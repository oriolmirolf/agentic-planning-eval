;; Problem 4: merge two towers into a single four‑block tower
;; Initial state: A sits on B; D sits on C.  Blocks A and D are clear.  B and C
;; rest on the table.  The arm is empty.
;; Goal: build a tower with A on B, B on C and C on D (so the final order from
;; top to bottom is A, B, C, D).

(define (problem p04)
  (:domain blocksworld)
  (:objects A B C D)
  (:init (arm-empty)
         (on A B) (on-table B)
         (on D C) (on-table C)
         (clear A) (clear D))
  (:goal (and (on A B) (on B C) (on C D)))
)