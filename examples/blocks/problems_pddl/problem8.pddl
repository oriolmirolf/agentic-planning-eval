;; Problem 8 (five blocks): create a single tower from three separate stacks
;; Initial state: B is on A; D is on C; E sits on the table alone.  Blocks B,
;; D and E are clear and the arm is empty.  Goal: build a five‑block tower with
;; C on D, D on B, B on A and A on E (from top to bottom: C, D, B, A, E).
;; This task is the most difficult because it requires merging three stacks into a
;; single five‑block tower.

(define (problem p10)
  (:domain blocksworld)
  (:objects A B C D E)
  (:init (arm-empty)
         (on B A) (on-table A)
         (on D C) (on-table C)
         (on-table E)
         (clear B) (clear D) (clear E))
  (:goal (and (on C D) (on D B) (on B A) (on A E)))
)