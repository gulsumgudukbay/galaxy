webpackJsonp([4],{116:function(e,i,t){"use strict";(function(e,i){var n=t(0),a=n,r=t(42).GalaxyApp,o=t(4),l=t(43);window.app=function(t,c){window.Galaxy=new r(t,c),Galaxy.debug("login app");var p=encodeURI(t.redirect);if(!t.show_welcome_with_login){var d=n.param({use_panels:"True",redirect:p});return void(window.location.href=Galaxy.root+"user/login?"+d)}var s=e.View.extend({initialize:function(i){this.page=i,this.model=new e.Model({title:o("Login required")}),this.setElement(this._template())},render:function(){this.page.$("#galaxy_main").prop("src",t.welcome_url)},_template:function(){return'<iframe src="'+t.root+"user/login?"+a.param({redirect:p})+'" frameborder="0" style="width: 100%; height: 100%;"/>'}});a(function(){Galaxy.page=new l.View(i.extend(t,{Right:s}))})}}).call(i,t(1),t(2))}},[116]);
//# sourceMappingURL=login.bundled.js.map