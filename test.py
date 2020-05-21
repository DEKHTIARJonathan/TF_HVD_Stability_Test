#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import sys
import unittest

from parameterized import parameterized
from parameterized import param

SUCCESS_CODE = 0


class HorovodTest(unittest.TestCase):

    def run_command(self, job_name, cmdline_flags):

        def exec_cmd(cmd):
            command_proc = subprocess.Popen(cmd)
            return_code = command_proc.wait()

            if return_code == SUCCESS_CODE:
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

        py_command = "python examples/%s.py %s" % (job_name, cmdline_flags)
        py_command = py_command.strip()
        run_command = py_command.split(" ")

        print("Command Executed: %s" % (" ".join(run_command)), file=sys.stderr)

        self.assertTrue(exec_cmd(run_command))

        return True

    @parameterized.expand([
        # tensorflow2_synthetic_benchmark
        param(
            "RN50 - Gradient Tape + HVD",
            filename="tensorflow2_synthetic_benchmark",
            cmdline_flags=""
        ),
        param(
            "RN50 - Gradient Tape + HVD + AMP",
            filename="tensorflow2_synthetic_benchmark",
            cmdline_flags="--use-amp"
        ),
        param(
            "RN50 - Gradient Tape + HVD + AMP + FP16 All Reduce",
            filename="tensorflow2_synthetic_benchmark",
            cmdline_flags="--use-amp --fp16-allreduce"
        ),
        # tensorflow2_mnist
        param(
            "keras.Sequential CTL - Gradient Tape + HVD",
            filename="tensorflow2_mnist",
            cmdline_flags=""
        ),
        param(
            "keras.Sequential CTL - Gradient Tape + HVD + AMP",
            filename="tensorflow2_mnist",
            cmdline_flags="--use-amp"
        ),
        param(
            "keras.Sequential CTL - Gradient Tape + HVD + AMP + FP16 All Reduce",
            filename="tensorflow2_mnist",
            cmdline_flags="--use-amp --fp16-allreduce"
        ),
        # tensorflow2_keras_mnist
        param(
            "keras fit & compile - Gradient Tape + HVD",
            filename="tensorflow2_keras_mnist",
            cmdline_flags=""
        ),
        param(
            "keras fit & compile - Gradient Tape + HVD + AMP",
            filename="tensorflow2_keras_mnist",
            cmdline_flags="--use-amp"
        ),
        param(
            "keras fit & compile - Gradient Tape + HVD + AMP + FP16 All Reduce",
            filename="tensorflow2_keras_mnist",
            cmdline_flags="--use-amp --fp16-allreduce"
        )
    ])
    def test_example(self, _, filename, cmdline_flags):
        self.run_command(job_name=filename, cmdline_flags=cmdline_flags)
