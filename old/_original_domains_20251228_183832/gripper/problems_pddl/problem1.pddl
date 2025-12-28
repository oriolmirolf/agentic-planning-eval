(define (problem gripper-p01)
  (:domain gripper)
  (:objects
    room1 room2 - room
    g1 g2 - gripper
    b1 - ball
  )
  (:init
    (= (total-cost) 0)
    (at-robot room1)
    (free g1)
    (free g2)
    (at b1 room1)
  )
  (:goal (and
    (at b1 room2)
  ))
  (:metric minimize (total-cost))
)
