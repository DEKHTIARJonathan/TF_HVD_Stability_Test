# TF_HVD_Stability_Test

```shell
$ pip install -r requirements.txt  # (assumed Tensorflow and Horovod are already installed)
$ pytest

=============================================== test session starts ===========================================
platform linux -- Python 3.6.9, pytest-5.4.1, py-1.8.1, pluggy-0.13.1 -- /usr/bin/python
cachedir: .pytest_cache
rootdir: /workspace, inifile: pytest.ini
collected 9 items    
                                                                                                                                                                                        
test.py::CustomTestCase::test_example_0_RN50_Gradient_Tape_HVD PASSED                                     [ 11%]
test.py::CustomTestCase::test_example_1_RN50_Gradient_Tape_HVD_AMP PASSED                                 [ 22%]
test.py::CustomTestCase::test_example_2_RN50_Gradient_Tape_HVD_AMP_FP16_All_Reduce PASSED                 [ 33%]
test.py::CustomTestCase::test_example_3_keras_Sequential_CTL_Gradient_Tape_HVD PASSED                     [ 44%]
test.py::CustomTestCase::test_example_4_keras_Sequential_CTL_Gradient_Tape_HVD_AMP PASSED                 [ 55%]
test.py::CustomTestCase::test_example_5_keras_Sequential_CTL_Gradient_Tape_HVD_AMP_FP16_All_Reduce PASSED [ 66%]
test.py::CustomTestCase::test_example_6_keras_fit_compile_Gradient_Tape_HVD PASSED                        [ 77%]
test.py::CustomTestCase::test_example_7_keras_fit_compile_Gradient_Tape_HVD_AMP PASSED                    [ 88%]
test.py::CustomTestCase::test_example_8_keras_fit_compile_Gradient_Tape_HVD_AMP_FP16_All_Reduce PASSED    [100%]

============================================== slowest test durations ==========================================

17.79s call     test.py::CustomTestCase::test_example_2_RN50_Gradient_Tape_HVD_AMP_FP16_All_Reduce
17.55s call     test.py::CustomTestCase::test_example_1_RN50_Gradient_Tape_HVD_AMP
16.00s call     test.py::CustomTestCase::test_example_0_RN50_Gradient_Tape_HVD
10.49s call     test.py::CustomTestCase::test_example_5_keras_Sequential_CTL_Gradient_Tape_HVD_AMP_FP16_All_Reduce
10.42s call     test.py::CustomTestCase::test_example_4_keras_Sequential_CTL_Gradient_Tape_HVD_AMP
8.62s call     test.py::CustomTestCase::test_example_3_keras_Sequential_CTL_Gradient_Tape_HVD
6.40s call     test.py::CustomTestCase::test_example_8_keras_fit_compile_Gradient_Tape_HVD_AMP_FP16_All_Reduce
6.39s call     test.py::CustomTestCase::test_example_7_keras_fit_compile_Gradient_Tape_HVD_AMP
6.33s call     test.py::CustomTestCase::test_example_6_keras_fit_compile_Gradient_Tape_HVD
(0.00 durations hidden.  Use -vv to show these durations.)

=========================================== 9 passed in 100.13s (0:01:40) =======================================
```
