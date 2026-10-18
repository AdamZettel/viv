# SPDX-License-Identifier: MIT OR Apache-2.0
# SPDX-FileCopyrightText: 2026 The Viv authors
"""`python -m vivkernel -f <connection file>`: how Jupyter starts the kernel."""

from ipykernel.kernelapp import IPKernelApp

from .kernel import VivKernel

IPKernelApp.launch_instance(kernel_class=VivKernel)
