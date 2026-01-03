;; Problem 5: UNSOLVABLE (Circular Goal)
;; Initial state: D is on C, C on B, B on A and A on the table.
;; Goal: A on B, B on C, and C on A (Circular dependency).

(define (problem p05)
  (:domain blocksworld)
  (:objects A B C D)
  (:init (arm-empty)
         (on-table A)
         (on B A)
         (on C B)
         (on D C)
         (clear D))
  ;; UNSOLVABLE MODIFICATION: Circular goal (A->B->C->A)
  (:goal (and (on A B) (on B C) (on C A)))
)