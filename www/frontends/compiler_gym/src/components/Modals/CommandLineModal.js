/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext } from "react";
import { Modal, Button, FormControl, Row, Col } from "react-bootstrap";
import ApiContext from "../../context/ApiContext";
import { getCommandLineArray } from "../../utils/Helpers";

const CommandLineModal = ({
  showModal,
  handleModal,
  title,
  startSession,
  reward,
  benchmark,
}) => {
  const { compilerGym } = useContext(ApiContext);
  const [commandLineText, setCommandLineText] = useState("");

  const actionsList =
    compilerGym.actions &&
    Object.entries(compilerGym.actions).map(([name, action_id]) => ({
      name,
      action_id: action_id.toString(),
      children: [],
    }));

  const startNewSession = async (e) => {
    e.preventDefault();
    try {
      await startSession(reward, getCommandLineArray(commandLineText, actionsList), benchmark);
      setCommandLineText("");
      handleModal();
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <Modal size="lg" show={showModal} onHide={handleModal}>
      <Modal.Header closeButton>
        <Modal.Title>{title}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Row className="align-items-center mt-2">
          <Col lg={11} md={11} xs={10}>
            <FormControl
              className="bg-dark text-white"
              aria-describedby="basic-addon1"
              type="text"
              placeholder="Enter command line."
              value={commandLineText}
              onChange={(e) => setCommandLineText(e.target.value)}
            />
          </Col>
          <Col lg={1} md={1} xs={1}>
            <Button variant="success">
              <i className="bi bi-cpu" onClick={startNewSession}></i>
            </Button>
          </Col>
        </Row>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="default" onClick={handleModal}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default CommandLineModal;

//opt -licm -sroa -post-inline-ee-instrument -float2int -instsimplify -pgo-memop-opt -dce -early-cse-memssa -inferattrs -strip-nondebug -flattencfg -loop-interchange -float2int -add-discriminators -loop-vectorize -aggressive-instcombine -prune-eh -gvn-hoist -loop-predication -irce -instcombine -globalsplit -pgo-memop-opt -lcssa -loop-guard-widening -mldst-motion -loop-simplifycfg -hotcoldsplit -loop-vectorize -forceattrs -rewrite-statepoints-for-gc -coro-elide -forceattrs -deadargelim -infer-address-spaces -float2int -deadargelim -coro-early -libcalls-shrinkwrap -loop-fusion -prune-eh -aggressive-instcombine -coro-early -loop-load-elim -elim-avail-extern -flattencfg -ipconstprop -lower-matrix-intrinsics -aggressive-instcombine -strip -memcpyopt -lcssa -irce -scalarizer -lcssa -add-discriminators -coro-cleanup -coro-elide -sink -tailcallelim -barrier -lower-guard-intrinsic -bdce -deadargelim -load-store-vectorizer -die -reassociate -post-inline-ee-instrument -attributor -simple-loop-unswitch -loop-guard-widening -canonicalize-aliases -sink -pgo-memop-opt -bdce -ee-instrument -tailcallelim -loop-versioning -alignment-from-assumptions -instsimplify -simplifycfg -add-discriminators -pgo-memop-opt -dce -loop-simplifycfg -inject-tli-mappings -licm -strip-dead-prototypes -irce -globaldce -pgo-memop-opt -lower-widenable-condition -alignment-from-assumptions -globaldce -slp-vectorizer -globaldce -alignment-from-assumptions -loop-sink -barrier -instcombine -correlated-propagation -ipsccp -loop-instsimplify -sccp -alignment-from-assumptions -partial-inliner -forceattrs -loop-distribute -loop-interchange -separate-const-offset-from-gep -newgvn -loop-instsimplify -globaldce -indvars -deadargelim -instsimplify -redundant-dbg-inst-elim -loop-idiom -licm -loop-unroll-and-jam -hotcoldsplit -mergereturn -partially-inline-libcalls -div-rem-pairs -sink -strip-debug-declare -infer-address-spaces -instnamer -inline -gvn -functionattrs -nary-reassociate -loop-reroll -barrier -prune-eh -loop-unroll-and-jam -functionattrs -post-inline-ee-instrument -partial-inliner -simple-loop-unswitch -mergefunc -rewrite-statepoints-for-gc -loop-distribute -instnamer -memcpyopt -loop-load-elim -strip-debug-declare -ee-instrument -insert-gcov-profiling -ee-instrument -sink -simplifycfg -memcpyopt -die -deadargelim -newgvn -loop-idiom -lowerinvoke -scalarizer -loop-simplifycfg -early-cse-memssa -partial-inliner -globalsplit -lower-matrix-intrinsics -loop-data-prefetch -coro-cleanup -nary-reassociate -dce -forceattrs -always-inline -strip-nondebug -lower-expect -callsite-splitting -prune-eh -bdce -loop-versioning-licm -always-inline -post-inline-ee-instrument -insert-gcov-profiling -loop-data-prefetch -sancov -loop-guard-widening -alignment-from-assumptions -functionattrs -sroa -strip-nondebug -consthoist -jump-threading -guard-widening -consthoist -speculative-execution -die -reassociate -forceattrs -adce -loop-data-prefetch -float2int -forceattrs -loop-distribute -bdce -loop-load-elim -indvars -sancov -slp-vectorizer -tailcallelim -lower-guard-intrinsic -irce -float2int -consthoist -strip-debug-declare -sancov -loop-simplify -memcpyopt -loop-simplify -guard-widening -ipconstprop -coro-cleanup -hotcoldsplit -strip-debug-declare -instcombine -dse -strip-debug-declare -simplifycfg input.bc -o output.bc
