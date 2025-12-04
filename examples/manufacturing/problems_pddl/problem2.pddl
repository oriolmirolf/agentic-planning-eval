(define (problem manufacturing-problem-2)
  (:domain manufacturing)

  (:objects
    p1 p2 - part
    cutter1 drill1 paintbooth1 - machine
  )

  (:init
    (raw p1)
    (raw p2)

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
      (painted p2)
    )
  )

  (:metric minimize (total-cost))
)

