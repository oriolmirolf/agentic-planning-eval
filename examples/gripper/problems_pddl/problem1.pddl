(define (problem gripper-p01)
  (:domain gripper)
  (:objects
    roo_bcabae roo_59c817 - room
    gri_32fce4 gri_42e92a - gripper
    bal_07e7cd - ball
  )
  (:init
    (= (total-cost) 0)
    (at-robot roo_bcabae)
    (free gri_32fce4)
    (free gri_42e92a)
    (at bal_07e7cd roo_bcabae)
  )
  (:goal (and
    (at bal_07e7cd roo_59c817)
  ))
  (:metric minimize (total-cost))
)
