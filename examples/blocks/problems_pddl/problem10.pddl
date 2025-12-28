;; Problem 10: merge two towers into a seven‑block tower
;; Initial state: there are seven blocks A–G.  Blocks A, B and C form a 3‑block
;; tower (C on B on A on the table).  Blocks D, E, F and G form a 4‑block tower
;; (G on F on E on D on the table).  Blocks C and G are clear and the arm is
;; empty.  Goal: create a single tower where A is on B, B on C, C on D, D on E,
;; E on F and F on G (so A is top and G is bottom).  This demonstrates a more
;; complex seven‑block restructuring.

(define (problem p10)
  (:domain blocksworld)
  (:objects A B C D E F G)
  (:init (arm-empty)
         ;; first tower (C on B on A)
         (on C B) (on B A) (on-table A)
         ;; second tower (G on F on E on D)
         (on G F) (on F E) (on E D) (on-table D)
         ;; clear blocks
         (clear C) (clear G))
  (:goal (and (on A B) (on B C) (on C D) (on D E) (on E F) (on F G)))
)