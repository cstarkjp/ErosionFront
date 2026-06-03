(for d in */; do; name="${d%/}"; pushd $d; cp time_slices.pdf "../../Combo/time_slices_$name.pdf"; popd; done;)
pdftk time_slices_test_2layer_ηul0p* time_slices_test_1layer_ηul1p0.pdf cat output combined.pdf
pdftk  time_slices_test_1layer_ηul1p0.pdf time_slices_test_2layer_ηul0p8.pdf time_slices_test_2layer_ηul0p6.pdf time_slices_test_2layer_ηul0p4.pdf time_slices_test_2layer_ηul0p2.pdf cat output combined.pdf
