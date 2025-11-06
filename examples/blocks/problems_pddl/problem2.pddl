;; Problem 2: re‑arrange blocks from a simple stack to a different tower
;; Initial state: B sits on A and C sits directly on the table.  Blocks B and C
;; are clear, while A is not.  The robot arm is empty.
;; Goal: build a new tower with A on C and B on A.

(define (problem p02)
  (:domain blocksworld)
  (:objects A B C)
  (:init (arm-empty)
         (on B A) (on-table A)
         (on-table C)
         (clear B) (clear C))
  (:goal (and (on A C) (on B A)))
)