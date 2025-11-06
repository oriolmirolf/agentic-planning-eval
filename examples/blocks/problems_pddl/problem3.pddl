;; Problem 3: reverse a three‑block tower (example from the UMBC lecture notes)
;; Initial state: C is on B, which is on A, which is on the table.  Block C is
;; clear and the arm is empty.  Goal: have A on B and B on C【789902751098195†L211-L227】.

(define (problem p03)
  (:domain blocksworld)
  (:objects A B C)
  (:init (arm-empty)
         (on-table A)
         (on B A)
         (on C B)
         (clear C))
  (:goal (and (on A B) (on B C)))
)