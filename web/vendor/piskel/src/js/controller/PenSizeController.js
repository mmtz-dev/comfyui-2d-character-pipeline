(function () {
  var ns = $.namespace("pskl.controller");

  // anim-pipeline: a range input and a "N px" label in place of upstream's four
  // SizePicker dots (1-4 px), so every size up to the service's maximum can be
  // chosen with the mouse. SizePicker itself is untouched; the settings panel's
  // grid pickers still use it.
  ns.PenSizeController = function () {};

  ns.PenSizeController.prototype.init = function () {
    var container = document.querySelector(".pen-size-container");
    this.range = container.querySelector(".pen-size-range");
    this.label = container.querySelector(".pen-size-value");
    this.range.max = String(pskl.app.penSizeService.getMaxPenSize());
    pskl.utils.Event.addEventListener(this.range, "input", this.onRangeInput_, this);
    // Hand focus back once a drag ends, so [ ] and the tool shortcuts work
    // again without a click on the canvas first.
    pskl.utils.Event.addEventListener(this.range, "change", function () {
      this.range.blur();
    }, this);

    $.subscribe(Events.PEN_SIZE_CHANGED, this.onPenSizeChanged_.bind(this));
    this.updateSelectedOption_();
  };

  ns.PenSizeController.prototype.onRangeInput_ = function () {
    pskl.app.penSizeService.setPenSize(parseInt(this.range.value, 10));
    this.updateSelectedOption_();
  };

  ns.PenSizeController.prototype.onPenSizeChanged_ = function (e) {
    this.updateSelectedOption_();
  };

  ns.PenSizeController.prototype.updateSelectedOption_ = function () {
    var size = pskl.app.penSizeService.getPenSize();
    this.range.value = String(size);
    this.label.textContent = size + " px";
  };
})();
