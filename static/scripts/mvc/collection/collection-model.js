"use strict";define(["mvc/dataset/dataset-model","mvc/base-mvc","utils/localization"],function(t,e,n){var i={defaults:{model_class:"DatasetCollectionElement",element_identifier:null,element_index:null,element_type:null},_mergeObject:function(t){return _.extend(t,t.object,{element_id:t.id}),delete t.object,t},constructor:function(t,e){t=this._mergeObject(t),this.idAttribute="element_id",Backbone.Model.apply(this,arguments)},parse:function(t,e){var n=t;return n=this._mergeObject(n)}},o=Backbone.Model.extend(e.LoggableMixin).extend(i).extend({_logNamespace:"collections"}),s=Backbone.Collection.extend(e.LoggableMixin).extend({_logNamespace:"collections",model:o,toString:function(){return["DatasetCollectionElementCollection(",this.length,")"].join("")}}),l=t.DatasetAssociation.extend(e.mixin(i,{url:function(){return this.has("history_id")?Galaxy.root+"api/histories/"+this.get("history_id")+"/contents/"+this.get("id"):(console.warn("no endpoint for non-hdas within a collection yet"),Galaxy.root+"api/datasets")},defaults:_.extend({},t.DatasetAssociation.prototype.defaults,i.defaults),_downloadQueryParameters:function(){return"?to_ext="+this.get("file_ext")+"&hdca_id="+this.get("parent_hdca_id")+"&element_identifier="+this.get("element_identifier")},constructor:function(t,e){this.debug("\t DatasetDCE.constructor:",t,e),i.constructor.call(this,t,e)},hasDetails:function(){return this.elements&&this.elements.length},toString:function(){return["DatasetDCE(",this.get("element_identifier"),")"].join("")}})),r=s.extend({model:l,toString:function(){return["DatasetDCECollection(",this.length,")"].join("")}}),c=Backbone.Model.extend(e.LoggableMixin).extend(e.SearchableModelMixin).extend({_logNamespace:"collections",defaults:{collection_type:null,deleted:!1},collectionClass:s,initialize:function(t,e){this.debug(this+"(DatasetCollection).initialize:",t,e,this),this.elements=this._createElementsModel(),this.on("change:elements",function(){this.log("change:elements"),this.elements=this._createElementsModel()})},_createElementsModel:function(){this.debug(this+"._createElementsModel",this.collectionClass,this.get("elements"),this.elements);var t=this.get("elements")||[];this.unset("elements",{silent:!0});var e=this;return _.each(t,function(t,n){_.extend(t,{parent_hdca_id:e.get("id")})}),this.elements=new this.collectionClass(t),this.elements},toJSON:function(){var t=Backbone.Model.prototype.toJSON.call(this);return this.elements&&(t.elements=this.elements.toJSON()),t},inReadyState:function(){var t=this.get("populated");return this.isDeletedOrPurged()||t},hasDetails:function(){return 0!==this.elements.length},getVisibleContents:function(t){return this.elements},parse:function(t,e){var n=Backbone.Model.prototype.parse.call(this,t,e);return n.create_time&&(n.create_time=new Date(n.create_time)),n.update_time&&(n.update_time=new Date(n.update_time)),n},delete:function(t){return this.get("deleted")?jQuery.when():this.save({deleted:!0},t)},undelete:function(t){return!this.get("deleted")||this.get("purged")?jQuery.when():this.save({deleted:!1},t)},isDeletedOrPurged:function(){return this.get("deleted")||this.get("purged")},searchAttributes:["name","tags"],toString:function(){return"DatasetCollection("+[this.get("id"),this.get("name")||this.get("element_identifier")].join(",")+")"}}),a=c.extend({collectionClass:r,toString:function(){return"List"+c.prototype.toString.call(this)}}),d=a.extend({toString:function(){return"Pair"+c.prototype.toString.call(this)}}),u=c.extend(e.mixin(i,{constructor:function(t,e){this.debug("\t NestedDCDCE.constructor:",t,e),i.constructor.call(this,t,e)},toString:function(){return["NestedDCDCE(",this.object?""+this.object:this.get("element_identifier"),")"].join("")}})),h=s.extend({model:u,toString:function(){return["NestedDCDCECollection(",this.length,")"].join("")}}),g=d.extend(e.mixin(i,{constructor:function(t,e){this.debug("\t NestedPairDCDCE.constructor:",t,e),i.constructor.call(this,t,e)},toString:function(){return["NestedPairDCDCE(",this.object?""+this.object:this.get("element_identifier"),")"].join("")}})),m=h.extend({model:g,toString:function(){return["NestedPairDCDCECollection(",this.length,")"].join("")}}),f=c.extend({collectionClass:m,toString:function(){return["ListPairedDatasetCollection(",this.get("name"),")"].join("")}}),C=a.extend(e.mixin(i,{constructor:function(t,e){this.debug("\t NestedListDCDCE.constructor:",t,e),i.constructor.call(this,t,e)},toString:function(){return["NestedListDCDCE(",this.object?""+this.object:this.get("element_identifier"),")"].join("")}})),D=h.extend({model:C,toString:function(){return["NestedListDCDCECollection(",this.length,")"].join("")}});return{ListDatasetCollection:a,PairDatasetCollection:d,ListPairedDatasetCollection:f,ListOfListsDatasetCollection:c.extend({collectionClass:D,toString:function(){return["ListOfListsDatasetCollection(",this.get("name"),")"].join("")}})}});
//# sourceMappingURL=../../../maps/mvc/collection/collection-model.js.map