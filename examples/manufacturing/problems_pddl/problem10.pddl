(define (problem manufacturing-problem-10)
  (:domain manufacturing)

  (:objects
    p1 p2 p3 p4 p5 p6 p7 p8 - part
    cutter1 cutter2 drill1 drill2 combo1 paintbooth1 paintbooth2 - machine
  )

  (:init
    (raw p1)
    (raw p2)
    (cut p3)
    (cut p4)
    (drilled p4)
    (cut p5)
    (drilled p5)
    (cut p6)
    (drilled p6)
    (painted p6)
    (raw p7)
    (cut p8)

    (available cutter1)
    (available cutter2)
    (available drill1)
    (available drill2)
    (available combo1)
    (available paintbooth1)
    (available paintbooth2)

    (can-cut cutter1)
    (can-cut cutter2)
    (can-cut combo1)
    (can-drill drill1)
    (can-drill drill2)
    (can-drill combo1)
    (can-paint paintbooth1)
    (can-paint paintbooth2)

    (= (total-cost) 0)
  )

  (:goal
    (and
      (painted p1)  ;; raw
      (drilled p2)  ;; raw -> cut+drill
      (painted p3)  ;; cut
      (painted p4)  ;; cut+drilled
      (painted p5)  ;; cut+drilled
      (cut p7)      ;; raw -> cut only
      ;; p6 and p8 irrelevant
    )
  )

  (:metric minimize (total-cost))
)
