# Candidate Sources

- dense neighbors
- sparse lexical hits
- entity overlap
- claim overlap
- prior memory hints
- bridge-rich cluster representatives

Argumentative corpora:

- prefer same-topic opposite-stance candidates over same-side overlap
- require topic consistency before adding broad comparison candidates
- treat generic debate vocabulary as weak evidence

Selection discipline:

- use `rank_candidate_priority.py` output as a soft ordering signal, not a hard verdict
- prefer candidates that expose new retrieval surfaces or reusable bridge terms
- keep some diversity across topic-family siblings when several candidates look equally strong
- do not let duplicate-style overlap dominate the bundle unless bridge gain is also high
