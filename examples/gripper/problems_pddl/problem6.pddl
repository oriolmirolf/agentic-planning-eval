(define (problem gripper-p06)
  (:domain gripper)
  (:objects
    room1 room2 - room
    g1 g2 - gripper
    b1 b2 b3 b4 b5 b6 - ball
  )
  (:init
    (= (total-cost) 0)
    (at-robot room1)
    (free g1)
    (free g2)
    (at b1 room1)
(at b2 room1)
(at b3 room1)
(at b4 room1)
(at b5 room1)
(at b6 room1)
  )
  (:goal (and
    (at b1 room2)
    (at b2 room2)
    (at b3 room2)
    (at b4 room2)
    (at b5 room2)
    (at b6 room2)
  ))
  (:metric minimize (total-cost))
)
