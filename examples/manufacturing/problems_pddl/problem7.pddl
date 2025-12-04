(define (problem manufacturing-problem-7)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 p4 p5 - part
    cutter1 drill1 drill2 paintbooth1 paintbooth2 - machine
  )

  (:init
    (raw p1)
    (cut p2)
    (cut p3)
    (drilled p3)
    (raw p4)
    (cut p5)

    (available cutter1)
    (available drill1)
    (available drill2)
    (available paintbooth1)
    (available paintbooth2)

    (can-cut cutter1)
    (can-drill drill1)
    (can-drill drill2)
    (can-paint paintbooth1)
    (can-paint paintbooth2)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)  ;; raw
      (painted p2)  ;; already cut
      (painted p3)  ;; cut+drilled
      (drilled p4)  ;; raw -> cut+drill
      ;; p5 irrelevant
    )
  )

  (:metric minimize (total-cost))
)
