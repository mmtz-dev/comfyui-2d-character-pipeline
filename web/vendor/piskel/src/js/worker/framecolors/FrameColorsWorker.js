(function () {
  var ns = $.namespace("pskl.worker.framecolors");

  if (Constants.TRANSPARENT_COLOR !== "rgba(0, 0, 0, 0)") {
    throw "Constants.TRANSPARENT_COLOR, please update FrameColorsWorker";
  }

  ns.FrameColorsWorker = function () {
    var getFrameColors = function (frame) {
      var frameColors = {};
      var transparentColorInt = this.TRANSPARENT_COLOR;
      var colors = 0;
      for (
        var i = 0, length = frame.length;
        i < length && colors < this.MAX_WORKER_COLORS;
        i++
      ) {
        var color = frame[i];
        if (color !== transparentColorInt) {
          if (!frameColors[color]) {
            frameColors[color] = true;
            colors++;
          }
        }
      }
      return frameColors;
    };

    this.onmessage = function (event) {
      try {
        this.TRANSPARENT_COLOR = event.data[0];
        this.MAX_WORKER_COLORS = event.data[1];
        var frame = event.data[2];
        var colors = getFrameColors(frame);
        this.postMessage({
          type: "SUCCESS",
          colors: colors
        });
      } catch (e) {
        this.postMessage({
          type: "ERROR",
          message: e.message
        });
      }
    };
  };
})();
