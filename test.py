#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import unittest

import gpu_util

from parameterized import parameterized
from parameterized import param

import pytest

SUCCESS_CODE = 0


class HorovodTest(unittest.TestCase):

    def run_command(self, job_name, cmdline_flags, num_gpus):

        def exec_cmd(cmd):
            command_proc = subprocess.Popen(cmd)
            return_code = command_proc.wait()

            print("RETURN CODE: %d" % return_code, file=sys.stderr)

            if return_code != SUCCESS_CODE:
                sys.tracebacklimit = 0
                stdout, stderr = command_proc.communicate()
                raise RuntimeError(
                    "\n##################################################\n"
                    "[*] STDOUT:{error_stdout}\n"
                    "[*] STERR:{error_stderr}\n"
                    "[*] command launched: `{command}`\n"
                    "##################################################\n".format(
                        error_stdout=stdout.decode("utf-8") if stdout is not None else "",
                        error_stderr=stderr.decode("utf-8") if stderr is not None else "",
                        command=" ".join(cmd)
                    )
                )

            return True

        if num_gpus == "1":
            py_command = "python examples/%s.py %s" % (job_name, cmdline_flags)
        else:
            py_command = "horovodrun -np {} python examples/{}.py {}".format(
                num_gpus,
                job_name,
                cmdline_flags
            )

        py_command = py_command.strip()
        run_command = py_command.split(" ")

        print("Command Executed: %s" % (" ".join(run_command)), file=sys.stderr)

        self.assertTrue(exec_cmd(run_command))

        return True

    @parameterized.expand([
        # tensorflow2_synthetic_benchmark
        param(
            "RN50 - Gradient Tape + HVD - 1GPU",
            filename="tf2_R50_CTL_GradientTape",
            cmdline_flags="",
            num_gpus=1
        ),
        param(
            "RN50 - Gradient Tape + HVD + AMP - 1GPU",
            filename="tf2_R50_CTL_GradientTape",
            cmdline_flags="--use-amp",
            num_gpus=1
        ),
        param(
            "RN50 - Gradient Tape + HVD + AMP + FP16 All Reduce - 1GPU",
            filename="tf2_R50_CTL_GradientTape",
            cmdline_flags="--use-amp --fp16-allreduce",
            num_gpus=1
        ),
        param(
            "RN50 - Gradient Tape + HVD - 2GPUs",
            filename="tf2_R50_CTL_GradientTape",
            cmdline_flags="",
            num_gpus=2
        ),
        param(
            "RN50 - Gradient Tape + HVD + AMP - 2GPUs",
            filename="tf2_R50_CTL_GradientTape",
            cmdline_flags="--use-amp",
            num_gpus=2
        ),
        param(
            "RN50 - Gradient Tape + HVD + AMP + FP16 All Reduce - 2GPUs",
            filename="tf2_R50_CTL_GradientTape",
            cmdline_flags="--use-amp --fp16-allreduce",
            num_gpus=2
        ),
        # tensorflow2_mnist
        param(
            "keras.Sequential CTL - Gradient Tape + HVD - 1GPU",
            filename="tf2_KerasSequential_CTL_GradientTape",
            cmdline_flags="",
            num_gpus=1
        ),
        param(
            "keras.Sequential CTL - Gradient Tape + HVD + AMP - 1GPU",
            filename="tf2_KerasSequential_CTL_GradientTape",
            cmdline_flags="--use-amp",
            num_gpus=1
        ),
        param(
            "keras.Sequential CTL - Gradient Tape + HVD + AMP + FP16 All Reduce - 1GPU",
            filename="tf2_KerasSequential_CTL_GradientTape",
            cmdline_flags="--use-amp --fp16-allreduce",
            num_gpus=1
        ),
        param(
            "keras.Sequential CTL - Gradient Tape + HVD - 2GPUs",
            filename="tf2_KerasSequential_CTL_GradientTape",
            cmdline_flags="",
            num_gpus=2
        ),
        param(
            "keras.Sequential CTL - Gradient Tape + HVD + AMP - 2GPUs",
            filename="tf2_KerasSequential_CTL_GradientTape",
            cmdline_flags="--use-amp",
            num_gpus=2
        ),
        param(
            "keras.Sequential CTL - Gradient Tape + HVD + AMP + FP16 All Reduce - 2GPUs",
            filename="tf2_KerasSequential_CTL_GradientTape",
            cmdline_flags="--use-amp --fp16-allreduce",
            num_gpus=2
        ),
        # tensorflow2_keras_mnist
        param(
            "keras fit & compile - Gradient Tape + HVD - 1GPU",
            filename="tf2_FitCompile_GradientTape",
            cmdline_flags="",
            num_gpus=1
        ),
        param(
            "keras fit & compile - Gradient Tape + HVD + AMP - 1GPU",
            filename="tf2_FitCompile_GradientTape",
            cmdline_flags="--use-amp",
            num_gpus=1
        ),
        param(
            "keras fit & compile - Gradient Tape + HVD + AMP + FP16 All Reduce - 1GPU",
            filename="tf2_FitCompile_GradientTape",
            cmdline_flags="--use-amp --fp16-allreduce",
            num_gpus=1
        ),
        param(
            "keras fit & compile - Gradient Tape + HVD - 2GPUs",
            filename="tf2_FitCompile_GradientTape",
            cmdline_flags="",
            num_gpus=2
        ),
        param(
            "keras fit & compile - Gradient Tape + HVD + AMP - 2GPUs",
            filename="tf2_FitCompile_GradientTape",
            cmdline_flags="--use-amp",
            num_gpus=2
        ),
        param(
            "keras fit & compile - Gradient Tape + HVD + AMP + FP16 All Reduce - 2GPUs",
            filename="tf2_FitCompile_GradientTape",
            cmdline_flags="--use-amp --fp16-allreduce",
            num_gpus=2
        ),
        param(
            "keras fit & compile - NaN Stay in Sync - No AMP - 2GPUs",
            filename="tf2_nan_FitCompile",
            cmdline_flags="",
            num_gpus=2
        ),
        param(
            "keras fit & compile - NaN Stay in Sync - With AMP - 2GPUs",
            filename="tf2_nan_FitCompile",
            cmdline_flags="--use_amp",
            num_gpus=2
        ),
        param(
            "keras CTL - NaN Stay in Sync - No AMP - 2GPUs",
            filename="tf2_nan_CTL",
            cmdline_flags="",
            num_gpus=2
        ),
        param(
            "keras CTL - NaN Stay in Sync - With AMP - 2GPUs",
            filename="tf2_nan_CTL",
            cmdline_flags="--use_amp",
            num_gpus=2
        ),
    ])
    def test_example(self, _, filename, cmdline_flags, num_gpus):
        if len(gpu_util.getGPUs()) < num_gpus:
            pytest.skip(
                'Insufficient number of GPUs available. Test is skipped.\n'
                'Available: {}.\n'
                'Requested: {}.\n'.format(len(gpu_util.getGPUs()), num_gpus)
            )
        self.run_command(
            job_name=filename,
            cmdline_flags=cmdline_flags,
            num_gpus=num_gpus,
        )
