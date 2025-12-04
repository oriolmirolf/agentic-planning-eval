(define (problem manufacturing-problem-4)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 p4 - part
    cutter1 cutter2 drill1 paintbooth1 - machine
  )

  (:init
    (raw p1)
    (raw p2)
    (raw p3)
    (raw p4)

    (available cutter1)
    (available cutter2)
    (available drill1)
    (available paintbooth1)

    (can-cut cutter1)
    (can-cut cutter2)
    (can-drill drill1)
    (can-paint paintbooth1)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)
      (painted p2)
      (drilled p3)
      (cut p4)
    )
  )

  (:metric minimize (total-cost))
)
