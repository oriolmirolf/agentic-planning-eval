;; Problem 5: reverse a four‑block tower
;; Initial state: D is on C, C on B, B on A and A on the table.  D is the top
;; block (clear) and the arm is empty.  Goal: reverse the tower so that A is on
;; B, B on C and C on D (top to bottom: A, B, C, D).

(define (problem p06)
  (:domain blocksworld)
  (:objects A B C D)
  (:init (arm-empty)
         (on-table A)
         (on B A)
         (on C B)
         (on D C)
         (clear D))
  (:goal (and (on A B) (on B C) (on C D)))
)