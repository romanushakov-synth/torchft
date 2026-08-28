# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest import TestCase

import torch

from torchft.utils import record_event


class TestRecordEvent(TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_returns_the_event_it_recorded(self) -> None:
        """Regression: it recorded an event and dropped it, so every caller got ``None``.

        Each call site sends the result to another process, which waits on it before reading the
        tensor the collective wrote. With ``None`` there is nothing to wait on, so the reading
        process races the writing one -- silently, and only when the collective is slow enough to
        lose, which is why it showed up across nodes and not on a single one.
        """
        event = record_event()

        self.assertIsNotNone(event, "record_event must return the event it recorded")
        self.assertIsInstance(event, torch.cuda.Event)
        # interprocess=True is what lets a peer process wait on it, and torch refuses to hand out
        # an IPC handle for an event that was never recorded.
        self.assertIsNotNone(event.ipc_handle())
        event.synchronize()
        self.assertTrue(event.query())
