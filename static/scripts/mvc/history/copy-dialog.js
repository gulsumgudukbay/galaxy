"use strict";define(["mvc/ui/ui-modal","mvc/ui/error-modal","utils/localization"],function(e,a,o){var t={defaultName:_.template("Copy of '<%- name %>'"),title:_.template(o("Copying history")+' "<%- name %>"'),submitLabel:o("Copy"),errorMessage:o("History could not be copied."),progressive:o("Copying history"),activeLabel:o("Copy only the active, non-deleted datasets"),allLabel:o("Copy all datasets including deleted ones"),anonWarning:o("As an anonymous user, unless you login or register, you will lose your current history ")+o("after copying this history. "),_template:_.template(["<% if( isAnon ){ %>",'<div class="warningmessage">',"<%- anonWarning %>",o("You can"),' <a href="/user/login">',o("login here"),"</a> ",o("or")," ",' <a href="/user/create">',o("register here"),"</a>.","</div>","<% } %>","<form>",'<label for="copy-modal-title">',o("Enter a title for the new history"),":","</label><br />",'<input id="copy-modal-title" class="form-control" style="width: 100%" value="<%= name %>" />','<p class="invalid-title bg-danger" style="color: red; margin: 8px 0px 8px 0px; display: none">',o("Please enter a valid history title"),"</p>","<% if( allowAll ){ %>","<br />","<p>",o("Choose which datasets from the original history to include:"),"</p>",'<input name="copy-what" type="radio" id="copy-non-deleted" value="copy-non-deleted" ','<% if( copyWhat === "copy-non-deleted" ){ print( "checked" ); } %>/>','<label for="copy-non-deleted"> <%- activeLabel %></label>',"<br />",'<input name="copy-what" type="radio" id="copy-all" value="copy-all" ','<% if( copyWhat === "copy-all" ){ print( "checked" ); } %>/>','<label for="copy-all"> <%- allLabel %></label>',"<% } %>","</form>"].join("")),_showAjaxIndicator:function(){var e='<p><span class="fa fa-spinner fa-spin"></span> '+this.progressive+"...</p>";this.modal.$(".modal-body").empty().append(e).css({"margin-top":"8px"})},dialog:function(e,t,l){function n(){var o=e.$("#copy-modal-title").val();if(o){var l="copy-all"===e.$('input[name="copy-what"]:checked').val();e.$("button").prop("disabled",!0),i._showAjaxIndicator(),t.copy(!0,o,l).done(function(e){r.resolve(e)}).fail(function(e,n,s){var d={name:o,copyAllDatasets:l};a.ajaxErrorModal(t,e,d,i.errorMessage),r.rejectWith(r,arguments)}).done(function(){p&&e.hide()})}else e.$(".invalid-title").show()}l=l||{};var i=this,r=jQuery.Deferred(),s=(l.nameFn||this.defaultName)({name:t.get("name")}),d=l.allDatasets?"copy-all":"copy-non-deleted",c=!!_.isUndefined(l.allowAll)||l.allowAll,p=!!_.isUndefined(l.autoClose)||l.autoClose;this.modal=e;var y=l.closing_callback;return e.show(_.extend(l,{title:this.title({name:t.get("name")}),body:$(i._template({name:s,isAnon:Galaxy.user.isAnonymous(),allowAll:c,copyWhat:d,activeLabel:this.activeLabel,allLabel:this.allLabel,anonWarning:this.anonWarning})),buttons:_.object([[o("Cancel"),function(){e.hide()}],[this.submitLabel,n]]),height:"auto",closing_events:!0,closing_callback:function(e){e&&r.reject({cancelled:!0}),y&&y(e)}})),e.$("#copy-modal-title").focus().select(),e.$("#copy-modal-title").on("keydown",function(e){13===e.keyCode&&(e.preventDefault(),n())}),r}},l=_.extend({},t,{defaultName:_.template("imported: <%- name %>"),title:_.template(o("Importing history")+' "<%- name %>"'),submitLabel:o("Import"),errorMessage:o("History could not be imported."),progressive:o("Importing history"),activeLabel:o("Import only the active, non-deleted datasets"),allLabel:o("Import all datasets including deleted ones"),anonWarning:o("As an anonymous user, unless you login or register, you will lose your current history ")+o("after importing this history. ")});return function(a,o){o=o||{};var n=window.parent.Galaxy.modal||new e.View({});return o.useImport?l.dialog(n,a,o):t.dialog(n,a,o)}});
//# sourceMappingURL=../../../maps/mvc/history/copy-dialog.js.map