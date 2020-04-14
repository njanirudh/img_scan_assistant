def run_in_parallel(args):
    return args[0].method(args[1])

myclass = MyClass()
method_args = [1,2,3,4,5,6]
args_map = [ (myclass, arg) for arg in method_args ]
pool = Pool()
pool.map(run_in_parallel, args_map)