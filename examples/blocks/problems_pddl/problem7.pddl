;; Problem 7 (five blocks): merge two towers and reverse their order
;; Initial state: block B is on A and C is on B (tower of three blocks).  E is
;; on D (second tower of two blocks).  Blocks C and E are clear.  Goal: build a
;; five‑block tower so that A is on B, B on C, C on D and D on E (top to
;; bottom: A, B, C, D, E).  This task is harder than the four‑block problems but
;; easier than problem 10 since only two towers must be merged.

(define (problem p09)
  (:domain blocksworld)
  (:objects A B C D E)
  (:init (arm-empty)
         (on B A) (on C B) (on-table A)
         (on E D) (on-table D)
         (clear C) (clear E))
  (:goal (and (on A B) (on B C) (on C D) (on D E)))
)