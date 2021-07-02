import { dateMixin, ModelBase } from "./ModelBase";

export class History extends dateMixin(ModelBase) {
    // not deleted
    get active() {
        return !this.deleted && !this.purged;
    }

    get hidItems() {
        return parseInt(this.hid_counter) - 1;
    }

    get totalItems() {
        return Object.keys(this.contents_active).reduce((result, key) => {
            const val = this.contents_active[key];
            return result + parseInt(val);
        }, 0);
    }

    get statusDescription() {
        const status = [];
        if (this.shared) {
            status.push("Shared");
        }
        if (this.importable) {
            status.push("Accessible");
        }
        if (this.published) {
            status.push("Published");
        }
        if (this.isDeleted) {
            status.push("Deleted");
        }
        if (this.purged) {
            status.push("Purged");
        }
        return status.join(", ");
    }
}

History.equals = function (a, b) {
    return JSON.stringify(a) == JSON.stringify(b);
};
