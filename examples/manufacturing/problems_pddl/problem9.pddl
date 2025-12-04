(define (problem manufacturing-problem-9)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 p4 p5 p6 p7 - part
    cutter1 cutter2 drill1 drill2 paintbooth1 paintbooth2 - machine
  )

  (:init
    (raw p1)
    (raw p2)
    (cut p3)
    (cut p4)
    (drilled p4)
    (cut p5)
    (drilled p5)
    (painted p5)
    (raw p6)
    (cut p7)

    (available cutter1)
    (available cutter2)
    (available drill1)
    (available drill2)
    (available paintbooth1)
    (available paintbooth2)

    (can-cut cutter1)
    (can-cut cutter2)
    (can-drill drill1)
    (can-drill drill2)
    (can-paint paintbooth1)
    (can-paint paintbooth2)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)
      (painted p2)
      (drilled p3)
      (painted p4)
      (drilled p6)
      ;; p5 and p7 irrelevant
    )
  )

  (:metric minimize (total-cost))
)
