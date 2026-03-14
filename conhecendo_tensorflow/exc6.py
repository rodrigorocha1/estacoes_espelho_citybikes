import tensorflow as tf

@tf.function
def foo(x):
  return x ** 2

writer=tf.summary.create_file_writer('logs/')
with writer.as_default():
  tf.summary.trace_on()
  foo(tf.Variable(1, name='foo1')) # define a unique name for the variable
  foo(tf.Variable(2, name='foo2'))
  tf.summary.trace_export("foo", step=0)