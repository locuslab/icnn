# Code from Repo SimonRamstedt/ddpg
# Heavily modified

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('copy', False, 'copy code folder to outdir')
flags.DEFINE_boolean('gdb', False, 'open gdb on error')


def run(main, outdir):
    script = os.path.abspath(sys.modules['__main__'].__file__)
    scriptdir, scriptfile = os.path.split(script)

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print("outdir: " + outdir)

    if FLAGS.copy:
        shutil.copytree(run_folder, path, symlinks=True, ignore=shutil.ignore_patterns('.*'))

    Executor(main, outdir).execute()


# register clean up before anybody else does
on_exit_do = []


def on_exit():
    if on_exit_do:
        on_exit_do[0]()
atexit.register(on_exit)


class Executor:

    def __init__(self, main, outdir):
        signal.signal(signal.SIGINT, self.on_kill)
        signal.signal(signal.SIGTERM, self.on_kill)
        on_exit_do.append(self.on_exit)

        self.main = main
        self.outdir = outdir

    def on_exit(self):
        elapsed = time.time() - self.t_start
        self.info['end_time'] = time.time()
        xwrite(self.outdir, self.info)
        print('Elapsed seconds: {}\n'.format(elapsed))

    def on_kill(self, *args):
        self.info['run_status'] = 'aborted'
        print("Experiment aborted")
        sys.exit(-1)

    def execute(self):
        """ execute locally """
        try:
            self.info = xread(self.outdir)
        except:
            self.info = {}

        self.t_start = time.time()

        try:
            self.info['start_time'] = self.t_start
            self.info['run_status'] = 'running'
            xwrite(self.outdir, self.info)

            self.main()

            self.info['run_status'] = 'finished'
        except Exception as e:
            self.on_error(e)

    def on_error(self, e):
        self.info['run_status'] = 'error'

        # construct meaningful traceback
        import traceback
        traceback.print_exc()
        import sys
        import code
        type, value, tb = sys.exc_info()
        tbs = []
        tbm = []
        while tb is not None:
            stb = traceback.extract_tb(tb)
            filename = stb[0][0]
            tdir, fn = os.path.split(filename)
            maindir = os.path.dirname(sys.modules['__main__'].__file__)
            if tdir == maindir:
                tbs.append(tb)
                tbm.append("{} : {} : {} : {}".format(fn, stb[0][1], stb[0][2], stb[0][3]))

            tb = tb.tb_next

        # print custom traceback
        print("\n\n- Experiment error traceback (use --gdb to debug) -\n")
        print("\n".join(tbm) + "\n")
        print("{}: {}\n".format(e.__class__.__name__, e))

        # enter interactive mode (i.e. post mortem)
        if FLAGS.gdb:
            print("\nPost Mortem:")
            for i in reversed(range(len(tbs))):
                print("Level {}: {}".format(i, tbm[i]))
                # pdb.post_mortem(tbs[i])
                frame = tbs[i].tb_frame
                ns = dict(frame.f_globals)
                ns.update(frame.f_locals)
                code.interact(banner="", local=ns)
                print("\n")


def xwrite(path, data):
    with open(path + '/ezex.json', 'w+') as f:
        json.dump(data, f)


def xread(path):
    with open(path + '/ezex.json') as f:
        return json.load(f)
