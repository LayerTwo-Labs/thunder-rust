use std::collections::BTreeSet;

use utoipa::openapi::{self, Ref, RefOr, Schema};

/// Get all component refs
trait ComponentRefs {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_;
}

impl<T> ComponentRefs for Box<T>
where
    T: ComponentRefs,
{
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        <T as ComponentRefs>::component_refs(self)
    }
}

impl<T> ComponentRefs for Vec<T>
where
    T: ComponentRefs,
{
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.iter().flat_map(|item| item.component_refs())
    }
}

impl<T> ComponentRefs for Option<T>
where
    T: ComponentRefs,
{
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.iter().flat_map(|item| item.component_refs())
    }
}

impl ComponentRefs for Ref {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        std::iter::once(self)
    }
}

impl<T> ComponentRefs for RefOr<T>
where
    T: ComponentRefs,
{
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        match self {
            RefOr::Ref(r) => Box::new(r.component_refs())
                as Box<dyn Iterator<Item = &Ref> + '_>,
            RefOr::T(t) => Box::new(t.component_refs()),
        }
    }
}

impl ComponentRefs for openapi::AllOf {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.items.component_refs()
    }
}

impl ComponentRefs for openapi::schema::AnyOf {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.items.component_refs()
    }
}

impl ComponentRefs for openapi::schema::ArrayItems {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        (match self {
            Self::False => None,
            Self::RefOrSchema(roschema) => Some(roschema.component_refs()),
        })
        .into_iter()
        .flatten()
    }
}

impl ComponentRefs for openapi::Array {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let items_refs = self.items.component_refs();
        let prefix_items_refs = self.prefix_items.component_refs();
        items_refs.chain(prefix_items_refs)
    }
}

impl<T> ComponentRefs for openapi::schema::AdditionalProperties<T>
where
    T: ComponentRefs,
{
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        (match self {
            Self::RefOr(ref_or) => Some(ref_or.component_refs()),
            Self::FreeForm(_) => None,
        })
        .into_iter()
        .flatten()
    }
}

impl ComponentRefs for openapi::Object {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let properties_refs = self
            .properties
            .values()
            .flat_map(|ref_or_schema| ref_or_schema.component_refs());
        let additional_properties_refs =
            self.additional_properties.component_refs();
        let property_names = self.property_names.component_refs();
        properties_refs
            .chain(additional_properties_refs)
            .chain(property_names)
    }
}

impl ComponentRefs for openapi::OneOf {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.items.component_refs()
    }
}

impl ComponentRefs for Schema {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        match self {
            Schema::AllOf(all_of) => Box::new(all_of.component_refs())
                as Box<dyn Iterator<Item = &Ref> + '_>,
            Schema::AnyOf(any_of) => Box::new(any_of.component_refs()),
            Schema::Array(array) => Box::new(array.component_refs()),
            Schema::Object(object) => Box::new(object.component_refs()),
            Schema::OneOf(oneof) => Box::new(oneof.component_refs()),
            _ => Box::new(std::iter::empty()),
        }
    }
}

impl ComponentRefs for openapi::example::Example {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        std::iter::empty()
    }
}

impl ComponentRefs for openapi::path::Parameter {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.schema.component_refs()
    }
}

impl ComponentRefs for openapi::Header {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.schema.component_refs()
    }
}

impl ComponentRefs for openapi::encoding::Encoding {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.headers
            .values()
            .flat_map(|header| header.component_refs())
    }
}

impl ComponentRefs for openapi::Content {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let schema_ref = self.schema.component_refs();
        let example_refs = self
            .examples
            .values()
            .flat_map(|ref_or_example| ref_or_example.component_refs());
        let encoding_refs = self
            .encoding
            .values()
            .flat_map(|encoding| encoding.component_refs());
        schema_ref.chain(example_refs).chain(encoding_refs)
    }
}

impl ComponentRefs for openapi::request_body::RequestBody {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.content
            .values()
            .flat_map(|content| content.component_refs())
    }
}

impl ComponentRefs for openapi::link::Link {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        std::iter::empty()
    }
}

impl ComponentRefs for openapi::Response {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let header_refs = self
            .headers
            .values()
            .flat_map(|header| header.component_refs());
        let content_refs = self
            .content
            .values()
            .flat_map(|content| content.component_refs());
        let link_refs = self
            .links
            .values()
            .flat_map(|ref_or_link| ref_or_link.component_refs());
        header_refs.chain(content_refs).chain(link_refs)
    }
}

impl ComponentRefs for openapi::Responses {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.responses
            .values()
            .flat_map(|ref_or_response| ref_or_response.component_refs())
    }
}

impl ComponentRefs for openapi::path::Operation {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let param_refs = self.parameters.component_refs();
        let request_body_refs = self.request_body.component_refs();
        let responses_refs = self.responses.component_refs();
        param_refs.chain(request_body_refs).chain(responses_refs)
    }
}

impl ComponentRefs for openapi::PathItem {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let param_refs = self.parameters.component_refs();
        let operation_refs = {
            let operations = [
                &self.get,
                &self.put,
                &self.post,
                &self.delete,
                &self.options,
                &self.head,
                &self.patch,
                &self.trace,
            ];
            operations
                .into_iter()
                .flat_map(|operation| operation.component_refs())
        };
        param_refs.chain(operation_refs)
    }
}

impl ComponentRefs for openapi::Paths {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        self.paths
            .values()
            .flat_map(|path_item| path_item.component_refs())
    }
}

impl ComponentRefs for openapi::Components {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let schemas_refs = self
            .schemas
            .values()
            .flat_map(|ref_or_schema| ref_or_schema.component_refs());
        let response_refs = self
            .responses
            .values()
            .flat_map(|ref_or_response| ref_or_response.component_refs());
        schemas_refs.chain(response_refs)
    }
}

impl ComponentRefs for openapi::OpenApi {
    fn component_refs(&self) -> impl Iterator<Item = &Ref> + '_ {
        let paths_refs = self.paths.component_refs();
        let component_refs = self.components.component_refs();
        paths_refs.chain(component_refs)
    }
}

// Check for errors within a schema.
// This is a WIP and may not cover all possible errors.
fn check_schema<T>() -> anyhow::Result<()>
where
    T: utoipa::OpenApi,
{
    let schema: openapi::OpenApi = <T as utoipa::OpenApi>::openapi();
    let component_ref_locations = BTreeSet::<&str>::from_iter(
        schema
            .component_refs()
            .map(|r#ref| r#ref.ref_location.as_str()),
    );
    let component_schemas =
        BTreeSet::<&str>::from_iter(schema.components.iter().flat_map(
            |components| components.schemas.keys().map(|s| s.as_str()),
        ));
    // TODO: check that there are no ref cycles here
    for ref_loc in &component_ref_locations {
        let Some(loc) = ref_loc.strip_prefix("#/components/schemas/") else {
            anyhow::bail!("Unexpected prefix in ref location: `{ref_loc}`");
        };
        if !component_schemas.contains(loc) {
            anyhow::bail!("Missing schema referenced as `{ref_loc}`")
        }
    }
    // Check for redundant components
    for component in component_schemas {
        let component_ref = format!("#/components/schemas/{component}");
        if !component_ref_locations.contains(component_ref.as_str()) {
            anyhow::bail!("No references to {component_ref}")
        }
    }
    Ok(())
}

#[test]
fn check_schemas() -> anyhow::Result<()> {
    let () = check_schema::<crate::node::PrivateRpcDoc>()?;
    let () = check_schema::<crate::node::get_block::RpcDoc>()?;
    let () = check_schema::<crate::node::RpcDoc>()?;
    let () = check_schema::<crate::wallet::RpcDoc>()?;
    Ok(())
}
