;; Problem 1: build a three‑block tower from three blocks on the table
;; Initial state: A, B and C are each on the table and clear with an empty arm
;; Goal: stack A on B and B on C【789902751098195†L118-L198】.

(define (problem p01)
  (:domain blocksworld)
  (:objects A B C)
  (:init (arm-empty)
         (on-table A) (on-table B) (on-table C)
         (clear A) (clear B) (clear C))
  (:goal (and (on A B) (on B C)))
)