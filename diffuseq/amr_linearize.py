"""
amr_linearize.py — turn DocAMR graphs into model-ready token strings plus graph edges.

Replaces `simplify_docamr` of the earlier prep script, which dropped brackets (review bug B3),
stripped `:ARGn` digits in "simple" mode (B2) and leaked undefined variable names (B9).

Pipeline for one document:
    1. sanitize_docamr(text)        repair concepts the parser emits but PENMAN cannot read
    2. parse_document(text)         -> list of sentence trees in :snt1..:sntN order
    3. linearize_sentences(trees)   -> LinearizedAMR(words, triples) for one chunk of sentences
    4. map_words_to_tokens(...)     -> token position of each word inside a tokenized string
    5. triples_to_token_graph(...)  -> [[head_pos, dep_pos, label, label_pos], ...] for the dataset

Linearized form (bracketed PENMAN without variables):
    (s / sing-01 :ARG0 (i / i))  ->  "( sing-01 :ARG0 i )"
    - a node with children is wrapped in "( ... )"; a leaf is its bare concept;
    - a re-entrant variable is replaced by its concept; the edge points at the defining occurrence;
    - a reference to a variable not defined in the chunk is dropped with its relation (cross-sentence
      references in chunked documents);
    - quoted constants lose their quotes ("U.S." -> U.S.);
    - drop_sense=True removes the sense suffix of ordinary predicates (grow-01 -> grow) and keeps
      special -9x frames (have-org-role-91).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import penman

logging.getLogger("penman").setLevel(logging.ERROR)

ROLE_INSTANCE = ":instance"
SPECIAL_FRAME_RE = re.compile(r"-9\d$")       # have-org-role-91, include-91, rate-entity-91, ...
SENSE_RE = re.compile(r"-\d\d$")              # grow-01, look-02
DOC_VAR_RE = re.compile(r"^s\d+\.[A-Za-z0-9]+$")  # DocAMR variables: s12.p3


@dataclass
class LinearizedAMR:
    """Result of linearizing one chunk of sentence graphs.

    words:   whitespace-free items; " ".join(words) is the model input string.
    triples: (head_word, label_word, dep_word, label) — word indices into `words`.
             head = parent concept, label_word = position of the relation token, dep = child concept
             (the defining occurrence for re-entrant variables).
    """
    words: List[str] = field(default_factory=list)
    triples: List[Tuple[int, int, int, str]] = field(default_factory=list)
    var_word: Dict[str, int] = field(default_factory=dict)   # variable -> word index of its concept

    @property
    def text(self) -> str:
        return " ".join(self.words)


# --------------------------------------------------------------------------------------------------
# Step 1-2: sanitize and parse
# --------------------------------------------------------------------------------------------------

_CONCEPT_RE = re.compile(r"/ ([^():]*?)(?=\s*[():]|\s*$)")


def sanitize_docamr(text: str) -> str:
    """Repair concept strings that the parser emits but PENMAN cannot read.

    Observed in 4 of 1,144 documents: fraction concepts (`(s2.a / 1/3 ...`) and multi-word
    concepts (`(s65.u / Auf Wiedersehen)`). A '/' inside a concept becomes '∕' (U+2215) and a space
    becomes '_'; restore_concept() maps '∕' back to '/' at emission time.
    """
    def fix(m):
        concept = m.group(1).strip()
        if " " not in concept and "/" not in concept:
            return m.group(0)
        if concept.startswith('"'):
            return m.group(0)
        return "/ " + concept.replace("/", "∕").replace(" ", "_") + " "
    return _CONCEPT_RE.sub(fix, text)


def restore_concept(concept: str) -> str:
    return concept.replace("∕", "/")


def parse_document(text: str) -> List[penman.Tree]:
    """Parse one DocAMR document into sentence trees, ordered by their :sntN number.

    A document whose root is not `document` (single-sentence files) is returned as one tree.
    Raises penman.DecodeError when the document cannot be parsed even after sanitizing.
    """
    tree = penman.parse(sanitize_docamr(text))
    var, branches = tree.node
    concept = next((t for r, t in branches if r == "/"), None)
    if concept != "document":
        return [tree]
    sentences = []
    for role, target in branches:
        m = re.fullmatch(r":snt(\d+)", role)
        if m and isinstance(target, tuple):
            sentences.append((int(m.group(1)), penman.Tree(target)))
    sentences.sort(key=lambda x: x[0])
    return [t for _, t in sentences]


# --------------------------------------------------------------------------------------------------
# Step 3: linearize
# --------------------------------------------------------------------------------------------------

def _collect_concepts(node, var_to_concept: Dict[str, str]) -> None:
    var, branches = node
    for role, target in branches:
        if role == "/":
            var_to_concept[var] = target
        elif isinstance(target, tuple):
            _collect_concepts(target, var_to_concept)


def _surface(concept: str, drop_sense: bool) -> str:
    concept = restore_concept(str(concept))
    if drop_sense and not SPECIAL_FRAME_RE.search(concept):
        concept = SENSE_RE.sub("", concept)
    return concept


def _constant_surface(value: str) -> str:
    value = str(value)
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    value = value.replace(" ", "_")
    return value if value else "_"


def _looks_like_variable(value: str) -> bool:
    return bool(DOC_VAR_RE.match(str(value)))


def linearize_sentences(trees: Sequence[penman.Tree], drop_sense: bool = True,
                        max_depth: Optional[int] = None) -> LinearizedAMR:
    """Linearize a chunk of sentence trees into one bracketed string.

    Sentences of a chunk are concatenated in order. A reference to a variable defined anywhere in
    the chunk is resolved (emitted as its concept, edge to the defining occurrence); a reference to
    a variable outside the chunk is dropped together with its relation.
    max_depth: expand children only for nodes at depth < max_depth (root = depth 0); deeper nodes are
    emitted as bare concepts. A node reached through `:name` is always expanded (its :opN strings
    are the entity's name). None = no limit. Used to truncate coreference antecedents.

    Example:
        (s / sing-01 :ARG0 (i / i) :ARG1-of (f / fast))
        words   = ["(", "sing", ":ARG0", "i", ":ARG1-of", "fast", ")"]      (drop_sense=True)
        triples = [(1, 2, 3, ":ARG0"), (1, 4, 5, ":ARG1-of")]
    """
    var_to_concept: Dict[str, str] = {}
    for t in trees:
        _collect_concepts(t.node, var_to_concept)

    out = LinearizedAMR()
    defining_word: Dict[str, int] = {}

    def walk(node, parent_word: Optional[int], label_word: Optional[int], label: Optional[str], depth: int):
        var, branches = node
        concept = next((t for r, t in branches if r == "/"), var)
        # Children that will be emitted: references to variables outside the chunk are removed first,
        # so a node left without children is emitted as a bare leaf ("go", not "( go )").
        children = [(r, t) for r, t in branches
                    if r != "/" and (isinstance(t, tuple) or t in var_to_concept or not _looks_like_variable(t))]
        if max_depth is not None and depth >= max_depth and label != ":name":
            children = []
        if children:
            out.words.append("(")
        my_word = len(out.words)
        out.words.append(_surface(concept, drop_sense))
        defining_word[var] = my_word
        if parent_word is not None:
            out.triples.append((parent_word, label_word, my_word, label))
        for role, target in children:
            if isinstance(target, tuple):
                rel_word = len(out.words)
                out.words.append(role)
                walk(target, my_word, rel_word, role, depth + 1)
            elif target in var_to_concept:
                # Re-entrancy: emit the concept again, edge to the defining occurrence (or to this
                # occurrence when the definition comes later in the chunk; fixed up below).
                rel_word = len(out.words)
                out.words.append(role)
                ref_word = len(out.words)
                out.words.append(_surface(var_to_concept[target], drop_sense))
                out.triples.append((my_word, rel_word, ("ref", target, ref_word), role))
            else:
                rel_word = len(out.words)
                out.words.append(role)
                leaf_word = len(out.words)
                out.words.append(_constant_surface(target))
                out.triples.append((my_word, rel_word, leaf_word, role))
        if children:
            out.words.append(")")

    for t in trees:
        walk(t.node, None, None, None, 0)

    # Resolve re-entrancy placeholders now that every defining occurrence is known.
    resolved = []
    for head, rel_word, dep, label in out.triples:
        if isinstance(dep, tuple):
            _, var, ref_word = dep
            dep = defining_word.get(var, ref_word)
        resolved.append((head, rel_word, dep, label))
    out.triples = resolved
    out.var_word = dict(defining_word)
    return out


# --------------------------------------------------------------------------------------------------
# Step 3b: cross-sentence coreference context
# --------------------------------------------------------------------------------------------------
# DocAMR links a mention to an earlier sentence with ":same-as <var>" (99.9% of cross-sentence links
# in the train split point backwards; 59% point 6+ sentences back). A chunk-level linearization drops
# those references (the variable is undefined in the chunk). The functions below recover them as a
# context segment appended after the chunk's AMR:
#
#     <chunk AMR> [SEP] :same-as ( <antecedent 1 subtree> ) :same-as ( <antecedent 2 subtree> ) ...
#
# plus one graph triple per link: mention concept -> ":same-as" token -> antecedent root concept.

CONTEXT_SEPARATOR = "[SEP]"


@dataclass
class CorefLink:
    mention_var: str          # chunk variable whose node carries the reference
    role: str                 # relation of the reference (":same-as" for almost all links)
    antecedent_var: str       # antecedent after following the :same-as chain
    antecedent_sentence: int  # position of the antecedent's sentence in the document


def index_document(doc_trees: Sequence[Optional[penman.Tree]]) -> Dict[str, Tuple[int, tuple]]:
    """Every variable defined in the document -> (sentence position, its node).

    A None entry (a dropped metadata sentence) keeps its position but contributes no variable, so it can
    never become an antecedent.
    """
    index: Dict[str, Tuple[int, tuple]] = {}

    def walk(node, pos):
        var, branches = node
        index.setdefault(var, (pos, node))
        for role, target in branches:
            if isinstance(target, tuple):
                walk(target, pos)

    for pos, t in enumerate(doc_trees):
        if t is not None:
            walk(t.node, pos)
    return index


def _earlier_reference(node, index, before: int):
    """First (role, var) of `node` that references a variable defined in a sentence < before."""
    for role, target in node[1]:
        if role != "/" and not isinstance(target, tuple) and target in index and index[target][0] < before:
            return role, target
    return None


def find_coref_links(chunk_trees: Sequence[penman.Tree], doc_trees: Sequence[penman.Tree], chunk_start: int,
                     follow_chain: bool = True, max_links: int = 4) -> List[CorefLink]:
    """
    Cross-sentence references from a chunk to earlier sentences of the same document.

    Steps:
      1. index every variable of the document: var -> (sentence position, node)
      2. walk the chunk trees in emission order; a branch whose target is a variable defined in a sentence
         before `chunk_start` is a link (mention = the node carrying the branch)
      3. follow_chain: from the antecedent, follow its own reference to an even earlier sentence until a
         node has none (cycle-guarded) — the earliest mention usually carries the name (:name ...)
      4. keep the first link per antecedent, at most max_links

    Example: sentence 3 "(s3.h / he :same-as s2.p)", sentence 2 "(s2.p / person :same-as s1.p)",
             sentence 1 "(s1.p / person :name (s1.n / name :op1 \"Alan\"))"
             -> [CorefLink(mention_var="s3.h", role=":same-as", antecedent_var="s1.p", antecedent_sentence=0)]
    """
    index = index_document(doc_trees)
    links: List[CorefLink] = []
    seen = set()

    def walk(node):
        var, branches = node
        for role, target in branches:
            if isinstance(target, tuple):
                continue
            if role != "/" and target in index and index[target][0] < chunk_start:
                ante, visited = target, {target}
                while follow_chain:
                    ref = _earlier_reference(index[ante][1], index, index[ante][0])
                    if ref is None or ref[1] in visited:
                        break
                    ante = ref[1]
                    visited.add(ante)
                if ante not in seen:
                    seen.add(ante)
                    links.append(CorefLink(var, role, ante, index[ante][0]))
        for role, target in branches:
            if isinstance(target, tuple):
                walk(target)

    for t in chunk_trees:
        walk(t.node)
    return links[:max_links]


def append_coref_context(main: LinearizedAMR, links: Sequence[CorefLink], doc_trees: Sequence[penman.Tree],
                         drop_sense: bool = True, max_depth: int = 2) -> LinearizedAMR:
    """
    Append the antecedent subtrees of `links` to a chunk linearization.

    Step by step (word indices):
      1. words = main.words + ["[SEP]"]                           (only when there is at least one link)
      2. per link: words += [role] + linearize(antecedent subtree, max_depth).words
         the antecedent's triples are shifted by the offset of its first word
      3. link triple: (main.var_word[mention_var], index of `role`, index of the antecedent root, role)
    Example: main "( see :ARG0 he )" with a link he -> (s1.p / person :name (s1.n / name :op1 "Alan"))
      words   = ( see :ARG0 he ) [SEP] :same-as ( person :name ( name :op1 Alan ) )
      triples = main triples + (8, 9, 11, ":name"), (11, 12, 13, ":op1") + link (3, 6, 8, ":same-as")
    """
    if not links:
        return main
    index = index_document(doc_trees)
    out = LinearizedAMR(words=list(main.words) + [CONTEXT_SEPARATOR], triples=list(main.triples),
                        var_word=dict(main.var_word))
    for link in links:
        if link.mention_var not in main.var_word:
            continue
        label_word = len(out.words)
        out.words.append(link.role)
        offset = len(out.words)
        sub = linearize_sentences([penman.Tree(index[link.antecedent_var][1])], drop_sense=drop_sense,
                                  max_depth=max_depth)
        out.words.extend(sub.words)
        out.triples.extend((h + offset, l + offset, d + offset, lab) for h, l, d, lab in sub.triples)
        out.triples.append((main.var_word[link.mention_var], label_word,
                            offset + sub.var_word[link.antecedent_var], link.role))
    return out


# --------------------------------------------------------------------------------------------------
# Step 4-5: words -> token positions -> dataset graph
# --------------------------------------------------------------------------------------------------

def word_char_starts(words: Sequence[str], offset: int = 0) -> List[int]:
    """Character start of each word in " ".join(words), shifted by `offset`.

    Example: words ["(", "sing", ":ARG0"] -> [0, 2, 7]; with offset 10 -> [10, 12, 17].
    """
    starts, pos = [], offset
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    return starts


def map_words_to_tokens(text: str, char_starts: Sequence[int], hf_tokenizer) -> List[Optional[int]]:
    """Token position (special tokens included, [CLS] = 0) of the first subword of each word.

    Uses the fast tokenizer's offset mapping instead of word indices: the BERT pre-tokenizer splits
    punctuation ("U.S." -> U . S .), so word indices of the whitespace split do not match the
    tokenizer's word ids (8.5% misaligned edge endpoints in the earlier prep script).
    A word whose start falls inside a truncated or missing span maps to None.

    Example: text "( sing :ARG0 i )", starts [0, 2, 7, 13, 15] -> [1, 2, 3, 4, 5].
    """
    enc = hf_tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    start_to_token = {}
    for idx, (s, e) in enumerate(enc["offset_mapping"]):
        if e > s and s not in start_to_token:
            start_to_token[s] = idx
    return [start_to_token.get(c) for c in char_starts]


def triples_to_token_graph(triples, word_to_token: Sequence[Optional[int]]) -> List[list]:
    """Convert word-index triples to token-position graph entries.

    Output entry: [head_pos, dep_pos, label, label_pos]; entries with any unmapped position are
    dropped. Positions refer to the tokenized string that `word_to_token` was computed on.
    """
    graph = []
    for head, rel_word, dep, label in triples:
        h, d, lp = word_to_token[head], word_to_token[dep], word_to_token[rel_word]
        if h is None or d is None or lp is None:
            continue
        graph.append([h, d, label, lp])
    return graph


def relation_labels(lin: LinearizedAMR) -> List[str]:
    """Relation labels used in one linearization (for building the relation vocabulary)."""
    return [label for _, _, _, label in lin.triples]


def special_frames(lin: LinearizedAMR) -> List[str]:
    """-9x frame concepts used in one linearization."""
    return [w for w in lin.words if SPECIAL_FRAME_RE.search(w) and not w.startswith(":")]
