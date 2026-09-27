(function () {
  var ns = $.namespace("pskl.service.pensize");

  var MIN_PENSIZE = 1;
  var MAX_PENSIZE = 128;
  // anim-pipeline: `[` / `]` walk this table rather than +-1, so the 1..128
  // range is a few presses: every size to 8, then fours to 32, sixteens to 128.
  var PENSIZE_STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 28, 32,
    48, 64, 80, 96, 112, 128];

  /**
   * Service to retrieve and modify the current pen size.
   */
  ns.PenSizeService = function () {
    this.size = MIN_PENSIZE;
  };

  ns.PenSizeService.prototype.init = function () {
    this.size = pskl.UserSettings.get(pskl.UserSettings.PEN_SIZE);

    var shortcuts = pskl.service.keyboard.Shortcuts;
    pskl.app.shortcutService.registerShortcut(
      shortcuts.MISC.INCREASE_PENSIZE,
      this.increasePenSize_.bind(this)
    );
    pskl.app.shortcutService.registerShortcut(
      shortcuts.MISC.DECREASE_PENSIZE,
      this.decreasePenSize_.bind(this)
    );
  };

  ns.PenSizeService.prototype.increasePenSize_ = function () {
    var size = this.size;
    var next = PENSIZE_STEPS.filter(function (step) { return step > size; })[0];
    this.setPenSize(next === undefined ? MAX_PENSIZE : next);
  };

  ns.PenSizeService.prototype.decreasePenSize_ = function () {
    var size = this.size;
    var lower = PENSIZE_STEPS.filter(function (step) { return step < size; });
    this.setPenSize(lower.length ? lower[lower.length - 1] : MIN_PENSIZE);
  };

  ns.PenSizeService.prototype.getMaxPenSize = function () {
    return MAX_PENSIZE;
  };

  ns.PenSizeService.prototype.getPenSize = function () {
    return this.size;
  };

  ns.PenSizeService.prototype.setPenSize = function (size) {
    if (this.isPenSizeValid_(size) && size != this.size) {
      this.size = size;
      pskl.UserSettings.set(pskl.UserSettings.PEN_SIZE, size);
      $.publish(Events.PEN_SIZE_CHANGED);
    }
  };

  ns.PenSizeService.prototype.isPenSizeValid_ = function (size) {
    if (isNaN(size)) {
      return false;
    }

    return size >= MIN_PENSIZE && size <= MAX_PENSIZE;
  };
})();
