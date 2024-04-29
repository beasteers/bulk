import { UpdateMode } from "core/enums"
import { entries } from "core/util/object"
import { logger } from "core/logging"
import * as p from "core/properties";
import { AjaxDataSource } from "models/sources";

export namespace GraphQLDataSource {
    export type Attrs = p.AttrsOf<Props>
    export type Props = AjaxDataSource.Props & {
        query: p.Property<string>;
        variables: p.Property<{ [key: string]: any } | null>;
    }
}
export interface GraphQLDataSource extends GraphQLDataSource.Attrs {}

export class GraphQLDataSource extends AjaxDataSource {
    override properties: GraphQLDataSource.Props;

    constructor(attrs?: Partial<GraphQLDataSource.Attrs>) {
        super(attrs);
    }

    static {
        this.define<GraphQLDataSource.Props>(({ String, Dict, Nullable }) => ({
            query: [String, ""],
            variables: [Nullable(Dict(String)), null],
        }));
    }

    prepare_request(): XMLHttpRequest {
        const xhr = new XMLHttpRequest()
        xhr.open(this.method, this.data_url, true)
        xhr.withCredentials = true
        xhr.setRequestHeader("Content-Type", this.content_type)
    
        for (const [name, value] of entries(this.http_headers)) {
          xhr.setRequestHeader(name, value)
        }
    
        return xhr
      }

    get_data(mode: UpdateMode, max_size: number | null = null, _if_modified: boolean = false): void {
        const xhr = this.prepare_request()
        xhr.addEventListener("load", () => this.do_load(xhr, mode, max_size ?? undefined))
        xhr.addEventListener("error", () => this.do_error(xhr))
        xhr.send(JSON.stringify({ query: this.query, variables: this.variables || undefined }))
    }

    async do_load(xhr: XMLHttpRequest, mode: UpdateMode, max_size?: number): Promise<void> {
        if (xhr.status === 200) {
            const raw_data = JSON.parse(xhr.responseText)
            if(raw_data.errors) {
                logger.error(`Failed to fetch JSON for query ${this.query} with error: ${JSON.stringify(raw_data.errors, null, 2)}`)
            } else {
                await this.load_data(raw_data, mode, max_size)
            }
        }
    }
}