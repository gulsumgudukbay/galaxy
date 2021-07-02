import { DatasetCollection } from "../../model";
import { ContentProvider, processContentStreams } from "../ContentProvider";
import { collectionPayload } from "./collectionPayload";

export default {
    mixins: [ContentProvider],

    computed: {
        dsc() {
            if (this.parent instanceof DatasetCollection) {
                return this.parent;
            }
            return new DatasetCollection(this.parent);
        },
    },

    watch: {
        dsc(newDsc, oldDsc) {
            if (!(newDsc.id == oldDsc.id)) {
                this.resetScrollPos();
            }
        },
    },

    methods: {
        initStreams() {
            const { debouncePeriod, pageSize, params$, scrollPos$, debug } = this;
            const parent$ = this.watch$("dsc", true);
            const sources = { params$, parent$, scrollPos$ };
            const settings = { debouncePeriod, pageSize, debug };
            return processContentStreams(collectionPayload, sources, settings);
        },
    },
};
