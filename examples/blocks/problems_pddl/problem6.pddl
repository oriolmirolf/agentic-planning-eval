;; Problem 6: build a five‑block tower from scattered blocks
;; Initial state: A, B, C, D and E all sit individually on the table and are
;; clear.  The arm is empty.  Goal: stack the blocks to form the tower E on D on
;; C on B on A (so E is top and A is bottom).

(define (problem p06)
  (:domain blocksworld)
  (:objects A B C D E)
  (:init (arm-empty)
         (on-table A) (on-table B) (on-table C) (on-table D) (on-table E)
         (clear A) (clear B) (clear C) (clear D) (clear E))
  (:goal (and (on E D) (on D C) (on C B) (on B A)))
)