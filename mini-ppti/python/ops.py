from enum import Enum


class Domain(Enum):
    HE = "he"
    MPC = "mpc"
    PLAIN = "plain"


class OpKind(Enum):
    INPUT = "input"
    OUTPUT = "output"

    MATMUL = "matmul"
    ADD = "add"
    MUL = "mul"
    SUM = "sum"

    RSQRT = "rsqrt"
    EXP = "exp"
    DIV = "div"
    CMP = "cmp"


LINEAR_OPS = {
    OpKind.MATMUL,
    OpKind.ADD,
    OpKind.MUL,
    OpKind.SUM,
}

NONLINEAR_OPS = {
    OpKind.RSQRT,
    OpKind.EXP,
    OpKind.DIV,
    OpKind.CMP,
}


def is_linear_op(kind: OpKind) -> bool:
    return kind in LINEAR_OPS


def is_nonlinear_op(kind: OpKind) -> bool:
    return kind in NONLINEAR_OPS


def default_domain_for_op(kind: OpKind) -> Domain:
    if kind in {OpKind.INPUT, OpKind.OUTPUT}:
        return Domain.PLAIN
    if is_linear_op(kind):
        return Domain.HE
    if is_nonlinear_op(kind):
        return Domain.MPC
    raise ValueError(f"Unsupported operation kind: {kind}")