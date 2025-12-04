(define (problem manufacturing-problem-1)
  (:domain manufacturing)

  (:objects
    p1 - part
    cutter1 drill1 paintbooth1 - machine
  )

  (:init
    (raw p1)

    (available cutter1)
    (available drill1)
    (available paintbooth1)

    (can-cut cutter1)
    (can-drill drill1)
    (can-paint paintbooth1)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)
    )
  )

  (:metric minimize (total-cost))
)
