#!/usr/bin/env python

# Modified from Jacob's Reified Context code

from optparse import OptionParser
parser = OptionParser()
parser.add_option("--name", type="string", dest="name")
parser.add_option("--compile", action="store_true", dest="compile", default=False)
parser.add_option("--run", action="store_true", dest="run", default=False)
parser.add_option("--seed", type=long, help="random seed", default=9189181171)
parser.add_option("-g", type="int", dest="memory", default=5)
parser.add_option("--java-help", action="store_true", dest="java_help", default=False)
parser.add_option("--tail", type="int", dest="tail", default=5)
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False)
parser.add_option("--dataset", type="string", dest="dataset", default="../data/a.txt")
parser.add_option("--iterations", type="int", dest="iterations", default=20)
parser.add_option("--inference", type="int", dest="inference", default=0)
parser.add_option("--fullySupervised", dest="fullySupervised", default=False)
parser.add_option("--debugVerbose", dest="debugVerbose", default=True)
parser.add_option("--stateVerbose", dest="stateVerbose", default=True)
parser.add_option("--generateData", default=False)

(options, args) = parser.parse_args()
if not options.name:
  print "No name given, defaulting to SCRATCH"
name = options.name or "SCRATCH"

from subprocess import call
from glob import glob
import shlex
import threading
import time
include="lib/fig.jar:lib/stanford-corenlp-3.5.1.jar"
prefix="state/execs"

if options.compile:
  call(["rm", "-f"] + glob("*.class"))
  call(["javac", "-cp", ".:%s" % include, "Main.java"])
  call(["mkdir", "-p", "classes/%s" % name])
  call(["mv"] + glob("*.class") + ["classes/%s/" % name])
  call(["mkdir", "-p", "%s/%s" % (prefix, name)])

if options.run:
  call_args = ["java", "-Xmx%dg" % options.memory, "-cp .:%s:classes/%s" % (include, name), 
               "Main", "-execPoolDir %s/%s" % (prefix, name)]
  if options.java_help:
    call_args.append("-help")
    call(shlex.split(" ".join(call_args)))
  else:
    if not options.verbose:
      call_args.append("-log.stdout true")
    call_args.append("-experimentName %s" % name)
    call_args.append("-model LinearChainCRF") # only have one model at the moment
    call_args.append("-dataSource %s" % options.dataset)
    call_args.append("-numIters %d" % options.iterations)
    call_args.append("-inferType %d" % options.inference)
    call_args.append("-fullySupervised %d" % options.fullySupervised)
    call_args.append("-debugVerbose %s" % options.debugVerbose)
    call_args.append("-stateVerbose %s" % options.stateVerbose)
    call_args.append("-generateData %s" % options.generateData)
    command_to_run = " ".join(call_args)
    print("command to run")
    print(command_to_run)
    run_cmd = lambda : call(shlex.split(command_to_run))
    run_cmd()
