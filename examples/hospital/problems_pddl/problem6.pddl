(define (problem hospital-p06)
  (:domain hospital)

  (:objects
    p_icu  p_surg  p_med - patient
    b_icu1 b_icu2  b_surg1 b_med1 - bed
    w_icu  w_surg  w_med  - ward
  )

  (:init
    ;; Ward requirements
    (patient-ward p_icu  w_icu)
    (patient-ward p_surg w_surg)
    (patient-ward p_med  w_med)

    ;; Beds
    (bed-ward b_icu1  w_icu)
    (bed-ward b_icu2  w_icu)
    (bed-ward b_surg1 w_surg)
    (bed-ward b_med1  w_med)

    ;; Current occupancy (cyclic misplacement):
    ;; ICU patient in surgery, surgery in med, med in ICU.
    (in-bed p_icu  b_surg1)
    (in-bed p_surg b_med1)
    (in-bed p_med  b_icu1)

    ;; Only extra bed is ICU2, but it starts dirty.
    (bed-dirty b_icu2)

    (= (total-cost) 0)
  )

  ;; Target: each patient in the single bed of their ward.
  (:goal
    (and
      (in-bed p_icu  b_icu1)
      (in-bed p_surg b_surg1)
      (in-bed p_med  b_med1)
    )
  )

  (:metric minimize (total-cost))
)
