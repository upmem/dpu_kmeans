# -*- coding: utf-8 -*-
import dpu_kmeans as m


def test_add():
    assert m.add(1, 2) == 3


def test_sub():
    assert m.subtract(1, 2) == -1


def test_checksum():
    assert m.test_checksum() == "0x007f8000"
