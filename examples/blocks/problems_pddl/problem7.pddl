;; Problem 7: UNSOLVABLE (Spatial Contradiction)
;; Initial state: Tower B-A, Tower E-D.
;; Goal: A is on B AND A is on C. Impossible.

(define (problem p07)
  (:domain blocksworld)
  (:objects A B C D E)
  (:init (arm-empty)
         (on B A) (on C B) (on-table A)
         (on E D) (on-table D)
         (clear C) (clear E))
  ;; UNSOLVABLE MODIFICATION: A cannot be on B and on C at the same time.
  (:goal (and (on A B) (on A C) (on C D) (on D E)))
)