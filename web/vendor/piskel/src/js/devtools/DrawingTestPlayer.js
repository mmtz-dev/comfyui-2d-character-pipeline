(function () {
  var ns = $.namespace("pskl.devtools");

  ns.DrawingTestPlayer = function (testRecord, step) {
    this.initialState = testRecord.initialState;
    this.events = testRecord.events;
    this.referencePng = testRecord.png;
    this.step =
      step || this.initialState.step || ns.DrawingTestPlayer.DEFAULT_STEP;
    this.callbacks = [];
    this.shim = null;
    this.performance = 0;
  };

  ns.DrawingTestPlayer.DEFAULT_STEP = 50;

  ns.DrawingTestPlayer.prototype.start = function () {
    this.setupInitialState_();
    this.createMouseShim_();

    // Override the main drawing loop to record the time spent rendering.
    this.loopBackup = pskl.app.drawingLoop.loop;
    pskl.app.drawingLoop.loop = function () {
      var before = window.performance.now();
      this.loopBackup.call(pskl.app.drawingLoop);
      this.performance += window.performance.now() - before;
    }.bind(this);

    this.regenerateReferencePng(
      function () {
        this.playEvent_(0);
      }.bind(this)
    );
  };

  ns.DrawingTestPlayer.prototype.setupInitialState_ = function () {
    var size = this.initialState.size;
    var piskel = this.createPiskel_(size.width, size.height);
    pskl.app.piskelController.setPiskel(piskel);

    $.publish(Events.SELECT_PRIMARY_COLOR, [this.initialState.primaryColor]);
    $.publish(Events.SELECT_SECONDARY_COLOR, [
      this.initialState.secondaryColor
    ]);
    $.publish(Events.SELECT_TOOL, [this.initialState.selectedTool]);

    // Old tests do not have penSize stored in initialState, fallback to 1.
    pskl.app.penSizeService.setPenSize(this.initialState.penSize || 1);
  };

  ns.DrawingTestPlayer.prototype.createPiskel_ = function (width, height) {
    var descriptor = new pskl.model.piskel.Descriptor("TestPiskel", "");
    var piskel = new pskl.model.Piskel(width, height, 12, descriptor);
    var layer = new pskl.model.Layer("Layer 1");
    var frame = new pskl.model.Frame(width, height);

    layer.addFrame(frame);
    piskel.addLayer(layer);

    return piskel;
  };

  ns.DrawingTestPlayer.prototype.regenerateReferencePng = function (callback) {
    var image = new Image();
    image.onload = function () {
      this.referenceCanvas = pskl.utils.CanvasUtils.createFromImage(image);
      callback();
    }.bind(this);
    image.src = this.referencePng;
  };

  /**
   * Catch all mouse events to avoid perturbations during the test
   */
  ns.DrawingTestPlayer.prototype.createMouseShim_ = function () {
    this.shim = document.createElement("DIV");
    this.shim.style.cssText =
      "position:fixed;top:0;left:0;right:0;left:0;bottom:0;z-index:15000";
    this.shim.addEventListener(
      "mousemove",
      function (e) {
        e.stopPropagation();
        e.preventDefault();
      },
      true
    );
    document.body.appendChild(this.shim);
  };

  ns.DrawingTestPlayer.prototype.removeMouseShim_ = function () {
    this.shim.parentNode.removeChild(this.shim);
    this.shim = null;
  };

  /**
   * Determine the delay before playing the next event.
   * Keyboard events with Ctrl (undo/redo) and transform tool events trigger
   * async operations (history deserialization, resize) that need extra time.
   */
  ns.DrawingTestPlayer.prototype.getStepDelay_ = function (recordEvent) {
    if (!recordEvent) {
      return this.step;
    }
    // Undo/redo (Ctrl+Z, Ctrl+Y) trigger async history deserialization
    if (recordEvent.type === "keyboard-event" && recordEvent.event.ctrlKey) {
      return 1000;
    }
    // Transform tools (crop, rotate, etc.) may resize the piskel asynchronously
    if (recordEvent.type === "transformtool-event") {
      return 1000;
    }
    return this.step;
  };

  ns.DrawingTestPlayer.prototype.playEvent_ = function (index) {
    var recordEvent = this.events[index];
    var delay = this.getStepDelay_(recordEvent);

    this.timer = window.setTimeout(
      function () {
        // All events have already been replayed, finish the test.
        if (!recordEvent) {
          // Wait for async operations to settle before comparing results.
          this.waitForStableState_(
            function () {
              this.onTestEnd_();
            }.bind(this)
          );
          return;
        }

        var before = window.performance.now();
        if (recordEvent.type === "mouse-event") {
          this.playMouseEvent_(recordEvent);
        } else if (recordEvent.type === "keyboard-event") {
          this.playKeyboardEvent_(recordEvent);
        } else if (recordEvent.type === "color-event") {
          this.playColorEvent_(recordEvent);
        } else if (recordEvent.type === "tool-event") {
          this.playToolEvent_(recordEvent);
        } else if (recordEvent.type === "pensize-event") {
          this.playPenSizeEvent_(recordEvent);
        } else if (recordEvent.type === "transformtool-event") {
          this.playTransformToolEvent_(recordEvent);
        } else if (recordEvent.type === "instrumented-event") {
          this.playInstrumentedEvent_(recordEvent);
        } else if (recordEvent.type === "clipboard-event") {
          this.playClipboardEvent_(recordEvent);
        }

        // Record the time spent replaying the event
        this.performance += window.performance.now() - before;

        this.playEvent_(index + 1);
      }.bind(this),
      delay
    );
  };

  ns.DrawingTestPlayer.prototype.playMouseEvent_ = function (recordEvent) {
    var event = recordEvent.event;
    var screenCoordinates = pskl.app.drawingController.getScreenCoordinates(
      recordEvent.coords.x,
      recordEvent.coords.y
    );
    event.clientX = screenCoordinates.x;
    event.clientY = screenCoordinates.y;
    if (pskl.utils.UserAgent.isMac && event.ctrlKey) {
      event.metaKey = true;
    }

    if (event.type == "mousedown") {
      pskl.app.drawingController.onMousedown_(event);
    } else if (event.type == "mouseup") {
      pskl.app.drawingController.onMouseup_(event);
    } else if (event.type == "mousemove") {
      pskl.app.drawingController.onMousemove_(event);
    }
  };

  ns.DrawingTestPlayer.prototype.playKeyboardEvent_ = function (recordEvent) {
    var event = recordEvent.event;
    if (pskl.utils.UserAgent.isMac) {
      event.metaKey = event.ctrlKey;
    }

    event.preventDefault = function () {};
    pskl.app.shortcutService.onKeyDown_(event);
  };

  ns.DrawingTestPlayer.prototype.playColorEvent_ = function (recordEvent) {
    if (recordEvent.isPrimary) {
      $.publish(Events.SELECT_PRIMARY_COLOR, [recordEvent.color]);
    } else {
      $.publish(Events.SELECT_SECONDARY_COLOR, [recordEvent.color]);
    }
  };

  ns.DrawingTestPlayer.prototype.playToolEvent_ = function (recordEvent) {
    $.publish(Events.SELECT_TOOL, [recordEvent.toolId]);
  };

  ns.DrawingTestPlayer.prototype.playPenSizeEvent_ = function (recordEvent) {
    pskl.app.penSizeService.setPenSize(recordEvent.penSize);
  };

  ns.DrawingTestPlayer.prototype.playTransformToolEvent_ = function (
    recordEvent
  ) {
    pskl.app.transformationsController.applyTool(
      recordEvent.toolId,
      recordEvent.event
    );
  };

  ns.DrawingTestPlayer.prototype.playInstrumentedEvent_ = function (
    recordEvent
  ) {
    pskl.app.piskelController[recordEvent.methodName].apply(
      pskl.app.piskelController,
      recordEvent.args
    );
  };

  ns.DrawingTestPlayer.prototype.playClipboardEvent_ = function (recordEvent) {
    $.publish(recordEvent.event.type, {
      preventDefault: function () {},
      clipboardData: {
        items: [],
        setData: function () {}
      }
    });
  };

  /**
   * Poll until the piskel rendered output is stable (identical hash on two
   * consecutive checks). This ensures async operations like history
   * deserialization and relayout have completed before comparing results.
   */
  ns.DrawingTestPlayer.prototype.waitForStableState_ = function (
    callback,
    prevHash,
    attempts
  ) {
    attempts = attempts || 0;
    var renderer = new pskl.rendering.PiskelRenderer(pskl.app.piskelController);
    var canvas = renderer.renderAsCanvas();
    var data = canvas
      .getContext("2d")
      .getImageData(0, 0, canvas.width, canvas.height).data;
    var hash = data.length + ":" + data[0] + ":" + data[data.length - 1];
    for (var i = 0; i < data.length; i += 37) {
      hash += ":" + data[i];
    }

    if (hash === prevHash || attempts > 20) {
      callback();
    } else {
      window.setTimeout(
        function () {
          this.waitForStableState_(callback, hash, attempts + 1);
        }.bind(this),
        200
      );
    }
  };

  ns.DrawingTestPlayer.prototype.onTestEnd_ = function () {
    this.removeMouseShim_();
    // Restore the original drawing loop.
    pskl.app.drawingLoop.loop = this.loopBackup;

    // Retrieve the imageData corresponding to the spritesheet created by the test.
    var renderer = new pskl.rendering.PiskelRenderer(pskl.app.piskelController);
    var canvas = renderer.renderAsCanvas();
    var testData = canvas
      .getContext("2d")
      .getImageData(0, 0, canvas.width, canvas.height);

    // Retrieve the reference imageData corresponding to the reference data-url png stored for this test.
    var refCanvas = this.referenceCanvas;
    this.referenceData = refCanvas
      .getContext("2d")
      .getImageData(0, 0, refCanvas.width, refCanvas.height);

    // Compare the two imageData arrays.
    var success = true;
    for (var i = 0; i < this.referenceData.data.length; i++) {
      if (this.referenceData.data[i] != testData.data[i]) {
        success = false;
      }
    }

    $.publish(Events.TEST_RECORD_END, [success]);
    this.callbacks.forEach(
      function (callback) {
        callback({
          success: success,
          performance: this.performance
        });
      }.bind(this)
    );
  };

  ns.DrawingTestPlayer.prototype.addEndTestCallback = function (callback) {
    this.callbacks.push(callback);
  };
})();
