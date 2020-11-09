# Dataset Parameters

| | [More Dense, Dense] | [SemiDense, Sparse] |
|-|-------------------|-------------------|
| Block Size | 2km | 1km | 

---

| | [More Dense, Dense, SemiDense] | Sparse |
|-|-------------------|-------------------|
| Filter | 20 Ping, 15 User, 15 Avg User | 10 Ping, 2 User, 2 Avg User |

---

| Window/Interval | More Dense | Dense | SemiDense | Sparse |
|-|------------|-------|-----------|--------|
| Flatten | 180 / 180 | 120 / 120 | 180 / 180 | 60 / 60 |
| Rolling | 180 / 30  | 120 / 20  | 180 / 30  | 60 / 10 |

| Valid Data Index | More Dense | Dense | SemiDense | Sparse |
|-|------------|-------|-----------|--------|
| Flatten | [2: -6]  | [3: -9]  | [2: -6]  | [6: -18] |
| Rolling | [12: -36] | [18: -54] | [12: -36] | [36: -108] |

---

| Postfix | More Dense | Dense | SemiDense | Sparse |
|-|------------|-------|-----------|--------|
| Flatten | FDD | FD | FSemiD | FS |
| Rolling | RDD | RD | RSemiD | RS |

---
