(function() {
    var callWithJQuery;
  
    callWithJQuery = function(pivotModule) {
      if (typeof exports === "object" && typeof module === "object") {
        return pivotModule(require("jquery"));
      } else if (typeof define === "function" && define.amd) {
        return define(["jquery"], pivotModule);
      } else {
        return pivotModule(jQuery);
      }
    };
  
    callWithJQuery(function($) {
      return $.pivotUtilities.d3_renderers = {
        Treemap: function(pivotData, opts) {
          var addToTree, color, defaults, height, margin, result, rowKey, tree, treemap, value, width, _i, _len, _ref;
          defaults = {
            localeStrings: {}
          };
          opts = $.extend(defaults, opts);
          result = $("<div>").css({
            width: "100%",
            height: "100%"
          });
          tree = {
            name: "All",
            children: []
          };
          addToTree = function(tree, path, value) {
            var child, newChild, x, _i, _len, _ref;
            if (path.length === 0) {
              tree.value = value;
              return;
            }
            if (tree.children == null) {
              tree.children = [];
            }
            x = path.shift();
            _ref = tree.children;
            for (_i = 0, _len = _ref.length; _i < _len; _i++) {
              child = _ref[_i];
              if (!(child.name === x)) {
                continue;
              }
              addToTree(child, path, value);
              return;
            }
            newChild = {
              name: x
            };
            addToTree(newChild, path, value);
            return tree.children.push(newChild);
          };
          _ref = pivotData.getRowKeys();
          for (_i = 0, _len = _ref.length; _i < _len; _i++) {
            rowKey = _ref[_i];
            value = pivotData.getAggregator(rowKey, []).value();
            if (value != null) {
              addToTree(tree, rowKey, value);
            }
          }
          color = d3.scale.category10();
          width = $(window).width() / 1.4;
          height = $(window).height() / 1.4;
          margin = 10;
          treemap = d3.layout.treemap().size([width, height]).sticky(true).value(function(d) {
            return d.size;
          });
          d3.select(result[0]).append("div").style("position", "relative").style("width", (width + margin * 2) + "px").style("height", (height + margin * 2) + "px").style("left", margin + "px").style("top", margin + "px").datum(tree).selectAll(".node").data(treemap.padding([15, 0, 0, 0]).value(function(d) {
            return d.value;
          }).nodes).enter().append("div").attr("class", "node").style("background", function(d) {
            if (d.children != null) {
              return "lightgrey";
            } else {
              return color(d.name);
            }
          }).text(function(d) {
            return d.name;
          }).call(function() {
            this.style("left", function(d) {
              return d.x + "px";
            }).style("top", function(d) {
              return d.y + "px";
            }).style("width", function(d) {
              return Math.max(0, d.dx - 1) + "px";
            }).style("height", function(d) {
              return Math.max(0, d.dy - 1) + "px";
            });
          });
          return result;
        }
      };
    });
  
  }).call(this);
  
  //# sourceMappingURL=d3_renderers.js.map