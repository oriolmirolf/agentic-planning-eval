;; Problem 9: build a six‑block tower from scratch
;; Initial state: six blocks A–F are all on the table and clear.  The arm is
;; empty.  Goal: assemble them into a single tower with F on E, E on D, D on C,
;; C on B and B on A (so F is top and A is bottom).  This problem shows how
;; Blocksworld can scale beyond five blocks.

(define (problem p09)
  (:domain blocksworld)
  (:objects A B C D E F)
  (:init (arm-empty)
         (on-table A) (on-table B) (on-table C) (on-table D) (on-table E) (on-table F)
         (clear A) (clear B) (clear C) (clear D) (clear E) (clear F))
  (:goal (and (on F E) (on E D) (on D C) (on C B) (on B A)))
)