(define (problem bw-3)
  (:domain blocks)
  (:objects f h q - block)
  (:init
    (ontable f) (ontable h) (ontable q)
    (clear f) (clear h) (clear q)
    (handempty)
  )
  (:goal
    (and (on f h) (on h q))
  )
  (:metric minimize (total-cost))
)
