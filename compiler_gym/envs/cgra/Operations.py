
class Operation(object):
    def __init__(self, name, inputs, outputs, latency):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.latency = latency

    def __str__(self):
        return self.name

Operations = [
    # TODO --- should we support more operations as heterogeneous?
    # IMO most of the other things that are scheduled are
    # pretty vacuous, although we could explore supporting those.
    # Operation is: name, inputs, outputs, cycles.
    Operation("add", 2, 1, 1),
    Operation("mul", 2, 1, 1),
    Operation("sub", 2, 1, 1),
    Operation("div", 2, 1, 1),
    Operation("and", 2, 1, 1),
    Operation("or", 2, 1, 1),
    Operation("xor", 2, 1, 1),
    Operation("fmul", 2, 1, 1),
    Operation("fsub", 2, 1, 1),
    Operation("fadd", 2, 1, 1),
    Operation("fdiv", 2, 1, 1),
    Operation("rsh", 2, 1, 1),
    Operation("lsh", 2, 1, 1),
    Operation("load", 1, 1, 1),
    Operation("store", 1, 1, 1),
    Operation("const", 0, 1, 1),
    Operation("noop", 0, 0, 1),
]

def operation_index_of(op):
    ind = 0
    for e in Operations:
        if e.name == op.name:
            return ind
        else:
            print (e.name + " uneq " + str(op))
        ind += 1
    return -1

def operation_latency(op):
    # TODO --- model latency --- or at least expost this
    # to a configuration.
    return op.latency


def operation_from_name(n):
    ind = operation_index_of(n)
    return Operations[ind]