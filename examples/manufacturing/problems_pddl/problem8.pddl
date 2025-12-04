(define (problem manufacturing-problem-8)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 p4 p5 p6 - part
    cutter1 cutter2 drill1 combo1 paintbooth1 - machine
  )

  (:init
    (raw p1)
    (cut p2)
    (cut p3)
    (drilled p3)
    (raw p4)
    (raw p5)
    (raw p6)

    (available cutter1)
    (available cutter2)
    (available drill1)
    (available combo1)
    (available paintbooth1)

    (can-cut cutter1)
    (can-cut cutter2)
    (can-cut combo1)
    (can-drill drill1)
    (can-drill combo1)
    (can-paint paintbooth1)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)
      (painted p2)
      (painted p3)
      (drilled p4)
      (cut p5)
      ;; p6 irrelevant
    )
  )

  (:metric minimize (total-cost))
)
