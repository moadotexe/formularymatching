#AHO CORSICK 


# ac.py
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any

@dataclass(frozen=True)
class Hit:
    start: int
    end: int          # exclusive
    pat: str          # matched pattern (normalized)
    payload: Any      # e.g., {"pnf_key": "...", "source": "PNF"}

class Aho:
    def __init__(self, patterns_with_payloads: Iterable[Tuple[str, Any]]):
        # state 0 = root
        self.goto: List[Dict[str, int]] = [dict()]
        self.out:  List[List[int]]      = [[]]
        self.fail: List[int]            = [0]
        self.pats: List[str]            = []     # normalized patterns
        self.payloads: List[Any]        = []     # parallel to pats

        for p, payload in patterns_with_payloads:
            if not p: 
                continue
            self._insert(p)
            self.payloads.append(payload)

        self._build()

    def _insert(self, pat: str) -> None:
        s = 0
        for ch in pat:
            if ch not in self.goto[s]:
                self.goto[s][ch] = len(self.goto)
                self.goto.append(dict())
                self.out.append([])
                self.fail.append(0)
            s = self.goto[s][ch]
        self.out[s].append(len(self.pats))
        self.pats.append(pat)

    def _build(self) -> None:
        q = deque()
        for ch, s in self.goto[0].items():
            self.fail[s] = 0
            q.append(s)
        while q:
            r = q.popleft()
            for ch, s in self.goto[r].items():
                q.append(s)
                f = self.fail[r]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[s] = self.goto[f].get(ch, 0)
                self.out[s].extend(self.out[self.fail[s]])

    def finditer(self, text: str) -> Iterable[Hit]:
        s = 0
        for i, ch in enumerate(text):
            while s and ch not in self.goto[s]:
                s = self.fail[s]
            s = self.goto[s].get(ch, 0)
            if self.out[s]:
                for pid in self.out[s]:
                    pat = self.pats[pid]
                    yield Hit(i - len(pat) + 1, i + 1, pat, self.payloads[pid])
